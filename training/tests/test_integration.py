"""Training workflows using real cache readers, losses and Packed replay.

Synthetic full-shape states keep this independent of private captures. Only the
epoch count is shortened; these are pipeline checks, not quality experiments.
CUDA and real-publication checks require explicit opt-in environment settings.
Author: Winston Cai.
"""

import csv
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from training import load_runner, parse_layers, evaluate, losses, train
from training.models import read_policy, read_checkpoint
from training.data.batches import StateBatches
from training.data.dataset import (
    CaptureReader,
    file_sha256,
    load_cached_layer,
    _mapped_states,
    extend_coverage,
    preload_layer,
    validate_states,
)
from training.data.storage import audit, benchmark_batch_view
from training.losses import (
    forward_error,
    packed_rows,
    rounded_output,
    simulated_packed_rows,
)
from training.models import EvenConditionalScore, ExpandedConditionalScore
from training.train import fit


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="remnant-training-integration-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.addCleanup(
            setattr,
            torch.backends.cuda.matmul,
            "allow_tf32",
            torch.backends.cuda.matmul.allow_tf32,
        )
        self.addCleanup(
            setattr,
            torch.backends.cudnn,
            "allow_tf32",
            torch.backends.cudnn.allow_tf32,
        )
        self.addCleanup(
            torch.set_float32_matmul_precision, torch.get_float32_matmul_precision()
        )
        torch.set_num_threads(2)
        self.generator = torch.Generator().manual_seed(42)
        self.runner = (
            Path(__file__).resolve().parents[2]
            / "third_party/sglang/python/sglang/benchmark/remnant/calibrate.py"
        )
        self.math = load_runner(self.runner)

    def _state(self, role: str, ordinal: int) -> dict:
        """Reuse the original synthetic fixture with the cache's full geometry."""
        raw = torch.randn(512, 512, generator=self.generator)
        query = torch.randn(64, 512, generator=self.generator) * 0.1
        frequencies = torch.polar(
            torch.ones(512, 32), torch.randn(512, 32, generator=self.generator)
        )
        query_frequencies = torch.polar(
            torch.ones(32), torch.randn(32, generator=self.generator)
        )
        weight = torch.ones(512)
        swa = torch.randn(128, 512, generator=self.generator)
        sink = torch.zeros(64)
        rows = self.math.normalize(raw, weight, 1e-6, frequencies)
        output = self.math.attention(query, torch.cat((swa, rows)), sink, 0.1)[0]
        target = rounded_output(self.math, output, query_frequencies)
        return {
            "rid": role,
            "role": role,
            "ordinal": ordinal,
            "split": "held_out",
            "raw": raw,
            "weight": weight,
            "eps": 1e-6,
            "freq": frequencies,
            "query_freq": query_frequencies,
            "q": query,
            "swa": swa,
            "sink": sink,
            "scale": 0.1,
            "target": target,
            "target_unrounded": output,
            "reference_squared_norm": float(target.square().sum()),
        }

    def _cache(self, layer: int) -> Path:
        """Write an ordinary receipt-bound cache; use the real reader below."""
        cache = self.root / "cache"
        cache.mkdir()
        split = {"train": ["train"], "development": ["development"], "test": []}
        states = {
            role: [self._state(role, ordinal) for ordinal in range(count)]
            for role, count in (("train", 4), ("development", 10))
        }
        self.math.atomic_json(
            cache / "config.json",
            {
                "torch": "historical-cache-runtime",
                "split": split,
                "layers": [layer],
                "train_queries_per_issue": 4,
            },
        )
        config_digest = file_sha256(cache / "config.json")
        payload = cache / f"layer-{layer:02d}.pt"
        torch.save(states, payload)
        receipt = cache / f"layer-{layer:02d}.json"
        self.math.atomic_json(
            receipt,
            {
                "layer": layer,
                "config_sha256": config_digest,
                "states_sha256": file_sha256(payload),
                "identities": validate_states(states, split),
            },
        )
        self.math.atomic_json(
            cache / "index.json",
            {
                str(layer): {"record_sha256": file_sha256(receipt)},
            },
        )
        self.math.atomic_json(
            cache / "progress.json",
            {
                "status": "complete",
                "layers": [layer],
                "config_sha256": config_digest,
                "index_sha256": file_sha256(cache / "index.json"),
            },
        )
        return cache

    def _warm_policy(self, layer: int, *, rank16: bool = False) -> Path:
        """Supply an incumbent plus an untouched layer with a relative path."""
        selected = []
        for current in (8, 12):
            scorer_type = (
                EvenConditionalScore
                if rank16 and current == layer
                else ExpandedConditionalScore
            )
            scorer = scorer_type(torch.zeros(512), seed=42 + current)
            checkpoint = self.root / f"warm-layer-{current:02d}.json"
            self.math.atomic_json(
                checkpoint,
                {
                    "layer": current,
                    "selected_parameters": {
                        name: value.tolist()
                        for name, value in scorer.state_dict().items()
                    },
                },
            )
            selected.append(
                {
                    "layer": current,
                    "path": checkpoint.name,
                    "sha256": file_sha256(checkpoint),
                    "source": "incumbent",
                }
            )
        policy = self.root / "warm-policy.json"
        self.math.atomic_json(
            policy,
            {
                "selected": selected,
                "status": "development_audited",
                "selected_mean": 0.1,
            },
        )
        return policy

    def _train_arguments(
        self,
        cache: Path,
        policy: Path,
        recipe: str,
        layers: str,
        output: Path,
    ) -> list[str]:
        return [
            "--cache",
            str(cache),
            "--policy",
            str(policy),
            "--runner",
            str(self.runner),
            "--output",
            str(output),
            "--recipe",
            recipe,
            "--layers",
            layers,
            "--threads",
            "2",
        ]

    @staticmethod
    def _short_fit(*args, **kwargs):
        # Execute the real optimizer and all preflights; bound only duration.
        return fit(*args, **{**kwargs, "epochs": 2})

    @staticmethod
    def _policy_record(policy: Path, layer: int) -> tuple[dict, dict]:
        entry = read_policy(policy)[layer]
        return read_checkpoint(entry), entry

    def _assert_run_settings(self, settings: dict, cache: Path, policy: Path) -> None:
        self.assertEqual(settings["cache"], str(cache.resolve()))
        self.assertEqual(settings["policy"], str(policy.resolve()))
        self.assertEqual(settings["runner"], str(self.runner.resolve()))
        self.assertEqual(settings["torch"], str(torch.__version__))
        self.assertEqual(settings["device"], "cpu")
        self.assertEqual(settings["threads"], 2)

    def _train_and_evaluate(
        self,
        cache: Path,
        policy: Path,
        recipe: str,
        layer: int,
        *,
        train_batch_size: int = 8,
    ) -> tuple[Path, dict]:
        output = self.root / f"{recipe}-b{train_batch_size}"

        def smoke_fit(*args, **kwargs):
            return self._short_fit(
                *args, **{**kwargs, "train_batch_size": train_batch_size}
            )

        with patch("training.train.fit", side_effect=smoke_fit):
            run = train.main(
                self._train_arguments(cache, policy, recipe, str(layer), output)
            )
        checkpoint = output / f"selected-layer-{layer:02d}.json"
        self._assert_run_settings(
            json.loads((output / "config.json").read_text()), cache, policy
        )
        result = json.loads(checkpoint.read_text())
        self.assertTrue(json.loads((output / "progress.json").read_text())["complete"])
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(result["train_batch_size"], train_batch_size)
        if train_batch_size == 1:
            self.assertIsNone(result["batch_preflight"])
        else:
            self.assertIsNotNone(result["batch_preflight"])
        self.assertTrue(result["optimizer_reinitialized"])
        warm, _ = self._policy_record(policy, layer)
        self.assertNotEqual(
            result["current_parameters"]["theta"], warm["selected_parameters"]["theta"]
        )
        self.assertLessEqual(
            result["development_selected"]["mean"], result["development_warm"]["mean"]
        )
        self.assertLessEqual(
            result["development_selected"]["p95"], result["development_baseline"]["p95"]
        )
        partial = json.loads((output / f"partial-layer-{layer:02d}.json").read_text())
        self.assertEqual(partial["current_parameters"], result["current_parameters"])
        selected_policy = Path(run["policy"])
        self.assertEqual(selected_policy, output / "policy.json")
        published = json.loads(selected_policy.read_text())
        self.assertNotIn("parent_policy_sha256", published)
        self.assertEqual(published["trained_layers"], [layer])
        self.assertFalse(published["fresh_test_completed"])
        self.assertNotIn("selected_mean", published)
        self.assertNotIn("status", published)
        self.assertEqual([entry["layer"] for entry in published["selected"]], [8, 12])
        for entry in published["selected"]:
            if entry["layer"] == layer:
                self.assertEqual(entry["path"], checkpoint.name)
                self.assertNotIn("sha256", entry)
                self.assertNotIn("source", entry)
            else:
                old_record, old_entry = self._policy_record(policy, entry["layer"])
                new_record, new_entry = self._policy_record(
                    selected_policy, entry["layer"]
                )
                self.assertEqual(new_record, old_record)
                self.assertEqual(new_entry, old_entry)
                self.assertTrue(Path(entry["path"]).is_absolute())
                self.assertFalse((output / Path(entry["path"]).name).exists())
        report_path = self.root / f"{recipe}-b{train_batch_size}-evaluation.json"
        report = evaluate.main(
            [
                "--cache",
                str(cache),
                "--policy",
                str(selected_policy),
                "--runner",
                str(self.runner),
                "--output",
                str(report_path),
                "--layers",
                str(layer),
                "--threads",
                "2",
            ]
        )
        self.assertEqual(json.loads(report_path.read_text()), report)
        self._assert_run_settings(report["settings"], cache, selected_policy)
        development = report["layers"][0]["roles"]["development"]
        self.assertEqual(development["selected"], result["development_selected"])
        self.assertEqual(development["records"], result["selected_records"])
        self.assertFalse(report["fresh_test_completed"])
        return selected_policy, result

    def _workflow(self, recipe: str, layer: int) -> tuple[Path, Path, dict]:
        cache = self._cache(layer)
        warm_policy = self._warm_policy(layer, rank16=recipe == "rank64")
        for train_batch_size in (1, 8):
            with self.subTest(recipe=recipe, train_batch_size=train_batch_size):
                policy, result = self._train_and_evaluate(
                    cache, warm_policy, recipe, layer, train_batch_size=train_batch_size
                )
        return cache, policy, result

    def _published_capture(self) -> tuple[Path, Path, dict]:
        """Small real schema-2 bundles, using existing conversion/publication."""
        legacy, converted = self.root / "legacy", self.root / "converted"
        config = {
            "layers": [2, 4, 6],
            "requests": {
                rid: {"split": "held_out", "prompt_tokens": 100, "quota": 1}
                for rid in ("issue-a", "issue-b")
            },
        }
        self.math.atomic_json(legacy / "capture-config.json", config)
        shared_config = {**config, "storage_schema_version": 2, "conversion_only": True}
        self.math.atomic_json(converted / "capture-config.json", shared_config)
        native = torch.zeros(1, 584, dtype=torch.uint8)
        native[:, :448] = torch.ones(1, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        native[:, 448:576] = torch.ones(1, 64).to(torch.bfloat16).view(torch.uint8)
        native[:, 576:] = 127
        query, sink = torch.zeros(16, 512, dtype=torch.bfloat16), torch.zeros(16)
        output, _, lse = self.math.attention(
            query.float(), self.math.decode_native(native).repeat(2, 1), sink, 0.1
        )
        for rid in config["requests"]:
            for rank in range(4):
                for root in (legacy, converted):
                    self.math.atomic_json(
                        root / f"rank-{rank}" / rid / "reservoir.json", {"0": 0}
                    )
            for layer in config["layers"]:
                paths = []
                for rank in range(4):
                    path = (
                        legacy / f"rank-{rank}" / rid / f"layer-{layer:02d}-slot-00.pt"
                    )
                    torch.save(
                        {
                            "rank": rank,
                            "rid": rid,
                            "layer": layer,
                            "ordinal": 0,
                            "query_position": 1,
                            "norm_eps": 1e-6,
                            "softmax_scale": 0.1,
                            "csa_indices": torch.tensor([0], dtype=torch.int32),
                            "swa_indices": torch.tensor([0], dtype=torch.int32),
                            "csa_valid": torch.tensor([True]),
                            "swa_valid": torch.tensor([True]),
                            "raw_c": torch.ones(1, 512),
                            "raw_positions": torch.tensor([0]),
                            "csa_bytes": native,
                            "swa_bytes": native,
                            "norm_weight": torch.ones(512),
                            "csa_freqs": torch.ones(1, 32, dtype=torch.complex64),
                            "query_freqs": torch.ones(32, dtype=torch.complex64),
                            "q": query,
                            "sink": sink,
                            "bar_o": output.to(torch.bfloat16),
                            "o": output.to(torch.bfloat16),
                            "lse": lse,
                            "scheduler": {},
                        },
                        path,
                    )
                    paths.append(path)
                self.math.convert_shared_group(paths, converted)
            self.math.checkpoint_issue(
                converted,
                shared_config,
                {
                    "rid": rid,
                    "response_id": rid,
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 1},
                },
            )
        return legacy, converted, shared_config

    # Training and generated-policy handoff.
    def test_rank64_expansion_then_continuation(self):
        cache, policy, expanded = self._workflow("rank64", 8)
        _, continued = self._train_and_evaluate(cache, policy, "continuation", 8)
        self.assertEqual(
            continued["development_warm"], expanded["development_selected"]
        )
        self.assertEqual(
            continued["warm_theta"], expanded["selected_parameters"]["theta"]
        )
        self.assertEqual(continued["gradient_mode"], "straight_through")
        # Reuse the real fit above to isolate final handoff failure handling.
        atomic_json, link = self.math.atomic_json, os.link
        temporary_directory = tempfile.TemporaryDirectory
        for failure in (
            "stage", "progress", "link", "interrupt", "competitor", "hard-link", "cleanup"
        ):
            output = self.root / f"policy-{failure}"
            competing_bytes = []

            @contextmanager
            def staging_directory(*args, **kwargs):
                with temporary_directory(*args, **kwargs) as temporary:
                    yield temporary
                if failure == "cleanup":
                    raise OSError("policy scratch cleanup failed")

            def publish(path, value):
                if failure == "stage" and path.name == "policy.json":
                    raise OSError("policy staging failed")
                if (
                    failure == "progress"
                    and path == output / "progress.json"
                    and value["complete"]
                ):
                    raise OSError("complete progress write failed")
                return atomic_json(path, value)

            def publish_link(source, target):
                if failure == "link":
                    raise OSError("policy link failed")
                if failure == "competitor":
                    Path(target).write_text("another writer owns this")
                elif failure == "hard-link":
                    competing_bytes.append(Path(source).read_bytes())
                    link(source, target)
                link(source, target)
                if failure == "interrupt":
                    raise KeyboardInterrupt()

            with self.subTest(publication_failure=failure):
                with patch(
                    "training.train.load_runner", return_value=self.math
                ), patch.object(
                    self.math, "atomic_json", side_effect=publish
                ), patch("training.train.fit", return_value=dict(continued)), patch(
                    "training.train.os.link", side_effect=publish_link
                ), patch(
                    "training.train.tempfile.TemporaryDirectory", side_effect=staging_directory
                ):
                    with self.assertRaises(
                        KeyboardInterrupt if failure == "interrupt" else OSError
                    ):
                        train.main(
                            self._train_arguments(
                                cache, policy, "continuation", "8", output
                            )
                        )
                if failure == "competitor":
                    self.assertEqual(
                        (output / "policy.json").read_text(), "another writer owns this"
                    )
                elif failure == "hard-link":
                    self.assertEqual((output / "policy.json").read_bytes(), competing_bytes[0])
                else:
                    self.assertFalse((output / "policy.json").exists())
                self.assertFalse(
                    json.loads((output / "progress.json").read_text())["complete"]
                )
                self.assertEqual(list(output.glob(".policy-*")), [])
        # Both layers have bound receipts, but the second payload is missing.
        # Keep completed work if a later layer fails to load; binding receipts
        # must not require deserializing all layers before the first fit.
        config = json.loads((cache / "config.json").read_text())
        config["layers"] = [8, 12]
        self.math.atomic_json(cache / "config.json", config)
        digest = file_sha256(cache / "config.json")
        record = json.loads((cache / "layer-08.json").read_text())
        index = {}
        for layer in (8, 12):
            receipt = cache / f"layer-{layer:02d}.json"
            self.math.atomic_json(
                receipt, {**record, "layer": layer, "config_sha256": digest}
            )
            index[str(layer)] = {"record_sha256": file_sha256(receipt)}
        self.math.atomic_json(cache / "index.json", index)
        self.math.atomic_json(
            cache / "progress.json",
            {
                "status": "complete",
                "layers": [8, 12],
                "config_sha256": digest,
                "index_sha256": file_sha256(cache / "index.json"),
            },
        )
        incomplete = self.root / "incomplete"
        with patch("training.train.fit", side_effect=self._short_fit):
            with self.assertRaisesRegex(FileNotFoundError, "layer-12.pt"):
                train.main(
                    self._train_arguments(
                        cache, policy, "continuation", "8,12", incomplete
                    )
                )
        self.assertFalse((incomplete / "policy.json").exists())
        self.assertTrue((incomplete / "selected-layer-08.json").exists())
        self.assertTrue((incomplete / "partial-layer-08.json").exists())
        self.assertEqual(
            json.loads((incomplete / "progress.json").read_text()),
            {"layers": [8], "complete": False},
        )
        # Untrained entries must also be valid before any fitting starts.
        parent = json.loads(policy.read_text())
        duplicate = {
            **parent,
            "selected": parent["selected"] + [parent["selected"][-1]],
        }
        stale = {
            **parent,
            "selected": [
                {**entry, "path": "missing-checkpoint.json"} if entry["layer"] == 12 else entry
                for entry in parent["selected"]
            ],
        }
        for name, manifest, message in (
            ("duplicate", duplicate, "unique captured policy layers"),
            ("stale", stale, "cannot read policy checkpoint: layer 12"),
        ):
            invalid_policy = policy.with_name(f"{name}-policy.json")
            self.math.atomic_json(invalid_policy, manifest)
            rejected = self.root / name
            with self.assertRaisesRegex(ValueError, message):
                train.main(
                    self._train_arguments(
                        cache, invalid_policy, "continuation", "8", rejected
                    )
                )
            self.assertFalse(rejected.exists())
        # Parameter/layer structure remains required without a fingerprint.
        checkpoint = policy.parent / "selected-layer-08.json"
        record = json.loads(checkpoint.read_text())
        record["selected_parameters"] = None
        self.math.atomic_json(checkpoint, record)
        with self.assertRaisesRegex(ValueError, "complete selected checkpoint"):
            self._policy_record(policy, 8)

    def test_early_iteration_uses_current_code_without_provenance(self):
        cache = self._cache(8)
        config = json.loads((cache / "config.json").read_text())
        self.assertNotIn("numerical_source", config)
        self.assertNotEqual(config["torch"], str(torch.__version__))
        policy = self._warm_policy(8)
        manifest = json.loads(policy.read_text())
        for entry in manifest["selected"]:
            entry["sha256"] = "ignored-historical-fingerprint"
        self.math.atomic_json(policy, manifest)
        output = self.root / "early-iteration"
        with patch("training.train.fit", side_effect=self._short_fit):
            trained = train.main(
                self._train_arguments(cache, policy, "continuation", "8", output)
            )
        configuration = json.loads((output / "config.json").read_text())
        self.assertEqual(configuration["torch"], str(torch.__version__))
        self.assertFalse(any("sha256" in key for key in configuration))
        report = evaluate.main([
            "--cache", str(cache), "--policy", trained["policy"],
            "--runner", str(self.runner), "--output", str(self.root / "early.json"),
            "--layers", "8", "--threads", "2",
        ])
        self.assertNotIn("provenance", report)
        self.assertNotIn("source", report)
        self.assertEqual(report["settings"]["torch"], str(torch.__version__))

    def test_layer8_relaxed_training(self):
        _, _, result = self._workflow("relaxed", 8)
        self.assertEqual(result["gradient_mode"], "relaxed")

    def test_layer12_anchored_training(self):
        _, _, result = self._workflow("anchored", 12)
        self.assertEqual(result["gradient_mode"], "anchored")
        self.assertEqual(result["codec_anchor_policy"], "frozen_warm")
        self.assertEqual(result["codec_anchor_queries"], 4)
        self.assertFalse(result["codec_anchor_uses_development"])

    # Source identity and process lifetime.


    # Capture storage, coverage, and bounded transfer ownership.
    def test_storage_rejects_empty_cohort_and_layers(self):
        legacy, _, config = self._published_capture()
        for field, empty in (("requests", {}), ("layers", [])):
            self.math.atomic_json(
                legacy / "capture-config.json", {**config, field: empty}
            )
            for convert in (False, True):
                output = self.root / f"empty-{field}-{convert}"
                with self.subTest(field=field, convert=convert):
                    with self.assertRaisesRegex(
                        ValueError, "nonempty.*requests.*layers"
                    ):
                        audit(legacy, self.runner, output, convert=convert)
                    self.assertFalse((output / "storage.json").exists())

    def test_storage_missing_groups_retain_failed_diagnostics(self):
        legacy, _, config = self._published_capture()
        for rank in range(4):
            self.math.atomic_json(legacy / f"rank-{rank}/issue-b/reservoir.json", {})
        for convert in (False, True):
            output = self.root / f"missing-groups-{convert}"
            with self.subTest(convert=convert):
                with self.assertRaisesRegex(
                    ValueError, "missing capture groups.*issue-b"
                ):
                    audit(legacy, self.runner, output, convert=convert)
                partial = json.loads((output / "storage.json").read_text())
                self.assertEqual(partial["status"], "failed")
                self.assertEqual(partial["groups"], 3)
                self.assertEqual(partial["converted_groups"], 3 if convert else 0)
                self.assertEqual(partial["native_checks"], 3 if convert else 0)
                self.assertEqual(
                    partial["missing_groups"],
                    [{"rid": "issue-b", "layer": layer} for layer in config["layers"]],
                )
                with (output / "storage.csv").open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 6)

        # A missing layer must fail even when the same issue has other groups.
        paths = self.math.state_paths

        def omit_layer(root, config, rid, layer):
            if (rid, layer) == ("issue-a", 4):
                return []
            return paths(root, config, rid, layer)

        for rank in range(4):
            self.math.atomic_json(
                legacy / f"rank-{rank}/issue-b/reservoir.json", {"0": 0}
            )
        output = self.root / "missing-one-layer"
        with patch.object(self.math, "state_paths", side_effect=omit_layer), patch(
            "training.data.storage.load_runner", return_value=self.math
        ):
            with self.assertRaisesRegex(ValueError, "missing capture groups.*issue-a"):
                audit(legacy, self.runner, output)
        partial = json.loads((output / "storage.json").read_text())
        self.assertEqual(partial["groups"], 5)
        self.assertEqual(partial["missing_groups"], [{"rid": "issue-a", "layer": 4}])

    def test_reader_requires_complete_uncorrupted_publication(self):
        legacy, converted, config = self._published_capture()
        output = self.root / "reader-check"
        destination = output / "published"
        self.math.publish_issue(converted, destination, "issue-a")
        report_path = output / "training-reader-batched.json"
        with self.assertRaisesRegex(ValueError, "missing.*issue-b"):
            benchmark_batch_view(legacy, self.runner, output)
        self.assertFalse(report_path.exists())
        publication = self.math.publish_issue(converted, destination, "issue-b")
        report = benchmark_batch_view(legacy, self.runner, output)
        requested = sorted(config["requests"])
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["requested_issue_ids"], requested)
        self.assertEqual(report["issues"], len(requested))
        self.assertEqual([row["rid"] for row in report["records"]], requested)
        self.assertEqual(report["queries_compared"], 6)
        previous_report = report_path.read_bytes()
        with (destination / publication["bundle"]).open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "issue-b.*corrupt"):
            benchmark_batch_view(legacy, self.runner, output)
        self.assertEqual(report_path.read_bytes(), previous_report)


    def test_storage_accounts_for_shared_files_and_bounded_reading(self):
        legacy, converted, config = self._published_capture()
        report = audit(converted, self.runner, self.root / "storage-audit")
        payloads = list(converted.rglob("*.pt"))
        self.assertEqual(
            report["serialized_bytes"], sum(p.stat().st_size for p in payloads)
        )
        self.assertEqual(report["payload_files"], len(payloads))
        self.assertGreater(report["tensor_bytes_by_field"]["raw_c"], 0)
        self.assertEqual(report["converted_groups"], 0)
        self.assertFalse((self.root / "storage-audit/converted").exists())
        copied = audit(
            legacy, self.runner, self.root / "conversion-audit", convert=True
        )
        self.assertEqual(copied["converted_groups"], 6)
        self.assertTrue((self.root / "conversion-audit/converted").is_dir())

        destination = self.root / "published"
        for rid in config["requests"]:
            self.math.publish_issue(converted, destination, rid)
        reader = CaptureReader(self.runner, destination, sorted(config["requests"]))
        # Full archive hashes were already checked. Extraction must still check
        # member hashes, but must not hash that archive again for each view.
        digest = reader._runner.file_sha256
        calls = []

        def tracked(path):
            calls.append(Path(path))
            return digest(path)

        reader._runner.file_sha256 = tracked
        with patch.object(
            reader._runner, "load_state", wraps=reader._runner.load_state
        ) as loads:
            views = reader.load_views(self.root / "scratch", "issue-a", [2, 4, 6])
        self.assertEqual(loads.call_count, 3)
        self.assertEqual(set(views), {2, 4, 6})
        self.assertFalse(any(path.suffix == ".gz" for path in calls))
        self.assertEqual(list((self.root / "scratch").iterdir()), [])
        marker = destination / "published/issue-a.json"
        original = marker.read_bytes()
        marker.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            reader.load_views(self.root / "scratch", "issue-a", [2])
        marker.write_bytes(original)
        config_path = destination / "capture-config.json"
        original_config = config_path.read_bytes()
        config_path.write_bytes(original_config + b"\n")
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            reader.load_views(self.root / "scratch", "issue-a", [2])
        with self.assertRaisesRegex(ValueError, "different capture configuration"):
            CaptureReader(self.runner, destination, sorted(config["requests"]))
        config_path.write_bytes(original_config)
        for layers in ([2], [2, 4]):
            self.math.atomic_json(
                legacy / "capture-config.json", {**config, "layers": layers}
            )
            output = self.root / f"reader-{len(layers)}"
            # Reuse the already published fixture without another publication.
            (output / "published").parent.mkdir(parents=True, exist_ok=True)
            (output / "published").symlink_to(destination, target_is_directory=True)
            result = benchmark_batch_view(legacy, self.runner, output)
            self.assertEqual(result["records"][0]["layers"], layers)
        publication = json.loads(marker.read_text())
        with (destination / publication["bundle"]).open("ab") as stream:
            stream.write(b"changed after preflight")
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            reader.load_views(self.root / "scratch", "issue-a", [2])

        # Audited copies are nested in rank payloads and must count too.
        path = converted / "rank-1/issue-a/layer-02-slot-00.pt"
        snapshot = self.math.read_snapshot(path)
        collector = sys.modules[self.math.reference().__package__ + ".capture"]
        collector.write_shared_snapshot(converted, path, snapshot, 0, audit=True)
        audited = audit(converted, self.runner, self.root / "nested-audit")
        self.assertGreater(audited["tensor_bytes_by_field"]["audit_shared.raw_c"], 0)
        self.assertEqual(
            audited["serialized_bytes"], sum(p.stat().st_size for p in payloads)
        )
        # Logical field extents are not physical storage: Torch preserves aliases.
        for path in converted.glob("rank-*/*/*.pt"):
            record = torch.load(path, weights_only=True)
            record["o"] = record["bar_o"]
            torch.save(record, path)
        aliased = audit(converted, self.runner, self.root / "alias-audit")
        self.assertNotIn("serialization_overhead_bytes", aliased)
        self.assertEqual(
            aliased["serialized_bytes"], sum(p.stat().st_size for p in payloads)
        )
        self.assertEqual(
            aliased["logical_tensor_bytes"],
            sum(aliased["tensor_bytes_by_field"].values()),
        )

        # Two slots/issue exercise bounded preload without a full 50-issue fit.
        self.math.atomic_json(legacy / "capture-config.json", config)
        for rid in config["requests"]:
            receipt = {"finish_reason": "stop", "files_sha256": {}}
            for path in (legacy / f"rank-0/{rid}").glob("*.pt"):
                for rank in range(4):
                    source = legacy / f"rank-{rank}/{rid}" / path.name
                    duplicate = source.with_name(
                        source.name.replace("slot-00", "slot-01")
                    )
                    record = torch.load(source, weights_only=True)
                    record["ordinal"] = 1
                    torch.save(record, duplicate)
            for rank in range(4):
                self.math.atomic_json(
                    legacy / f"rank-{rank}/{rid}/reservoir.json", {"0": 0, "1": 1}
                )
                for path in (legacy / f"rank-{rank}/{rid}").iterdir():
                    receipt["files_sha256"][str(path.relative_to(legacy))] = (
                        file_sha256(path)
                    )
            self.math.atomic_json(legacy / f"done/{rid}.json", receipt)
        sizes = []
        load = self.math.load_state

        def observed(paths, device):
            sizes.append(len(self.math._snapshot_bytes))
            return load(paths, device)

        split = {"train": [], "development": ["issue-a", "issue-b"]}
        with patch.object(self.math, "load_state", side_effect=observed):
            states = preload_layer(
                self.math, legacy, config, split, 2, roles=("development",)
            )
        self.assertEqual(len(states["development"]), 4)
        self.assertEqual(sizes, [4, 4, 4, 4])
        self.assertEqual(self.math._snapshot_bytes, {})
        # Valid hashes cannot make a stale reservoir identity admissible.
        for key, wrong in (("ordinal", 7), ("rid", "wrong-issue"), ("layer", 4)):
            saved = {}
            receipt_path = legacy / "done/issue-a.json"
            receipt = json.loads(receipt_path.read_text())
            for rank in range(4):
                path = legacy / f"rank-{rank}/issue-a/layer-02-slot-00.pt"
                saved[path] = path.read_bytes()
                record = torch.load(path, weights_only=True)
                record[key] = wrong
                torch.save(record, path)
                receipt["files_sha256"][str(path.relative_to(legacy))] = (
                    file_sha256(path)
                )
            self.math.atomic_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "capture identity mismatch"):
                preload_layer(
                    self.math, legacy, config, split, 2, roles=("development",)
                )
            self.assertEqual(self.math._snapshot_bytes, {})
            for path, payload in saved.items():
                path.write_bytes(payload)
                receipt["files_sha256"][str(path.relative_to(legacy))] = (
                    file_sha256(path)
                )
            self.math.atomic_json(receipt_path, receipt)
        damaged = legacy / "rank-1/issue-a/layer-02-slot-00.pt"
        damaged.write_bytes(damaged.read_bytes() + b"bad")
        with self.assertRaisesRegex(ValueError, "capture payload changed"):
            preload_layer(self.math, legacy, config, split, 2, roles=("development",))
        self.assertEqual(self.math._snapshot_bytes, {})
        with patch("training.data.dataset.MAX_GROUP_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "rank group exceeds"):
                preload_layer(
                    self.math, legacy, config, split, 2, roles=("development",)
                )
        self.assertEqual(self.math._snapshot_bytes, {})
        # CUDA availability is not a reason to reject a CPU data operation.
        with patch("torch.cuda.is_available", return_value=True):
            with self.assertRaises(FileNotFoundError):
                extend_coverage(
                    legacy,
                    self.runner,
                    self.root / "missing-cache",
                    self.root / "extension",
                )

    def test_mapped_cache_and_bounded_anchor_training(self):
        cache = self._cache(12)
        payload = cache / "layer-12.pt"
        eager = torch.load(payload, map_location="cpu", weights_only=True)
        digest = file_sha256(payload)
        # Real reader and hash checks may not allocate a payload-sized byte string.
        with patch.object(
            Path, "read_bytes", side_effect=AssertionError("whole-file read")
        ):
            mapped = evaluate.load_cached_layer(cache, 12)
            self.assertEqual(file_sha256(payload), digest)
        for role, rows in eager.items():
            for expected, actual in zip(rows, mapped[role]):
                for key, value in expected.items():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(value, actual[key]), key)
                    else:
                        self.assertEqual(value, actual[key])
        legacy = self.root / "legacy.pt"
        torch.save(eager, legacy, _use_new_zipfile_serialization=False)
        with self.assertRaisesRegex(ValueError, "memory-map.*legacy.pt"):
            _mapped_states(legacy)

        # Nine states force an eight-state batch followed by a singleton.
        states = [dict(mapped["train"][index % 4], ordinal=index) for index in range(9)]
        stage = StateBatches.stage
        events, owners, vector_counts = [], [], []

        @contextmanager
        def observed(owner, indices, **kwargs):
            indices = tuple(int(index) for index in indices)
            with stage(owner, indices, **kwargs) as batch:
                self.assertLessEqual(owner.active_count, owner.limit)
                self.assertEqual(
                    [s["ordinal"] for s in batch.states],
                    [owner.states[index]["ordinal"] for index in indices],
                )
                self.assertTrue(all("target_unrounded" not in s for s in batch.states))
                if batch.anchors is not None:
                    self.assertTrue(
                        all(
                            a["state"] is s for a, s in zip(batch.anchors, batch.states)
                        )
                    )
                if owner.states is states:
                    events.append((indices, batch.anchors is not None))
                    if not owners:
                        owners.append(owner)
                yield batch
            self.assertEqual(owner.active_count, 0)
            self.assertEqual(batch.states, [])

        vector = losses.relaxed_forward_errors

        def counted_vector(math, batch, *args, **kwargs):
            vector_counts.append(batch["raw"].shape[0])
            return vector(math, batch, *args, **kwargs)

        with patch.object(StateBatches, "stage", observed), patch.object(
            losses, "codec_residual_anchor", wraps=losses.codec_residual_anchor
        ) as anchors, patch.object(
            losses, "relaxed_forward_errors", side_effect=counted_vector
        ):
            result = fit(
                self.math,
                self.math.reference(),
                states,
                mapped["development"],
                torch.zeros(512),
                mode="conditional_expanded",
                epochs=2,
                train_batch_size=8,
                gradient_mode="anchored",
                device="cpu",
                scratch_root=self.root,
            )
        self.assertEqual(anchors.call_count, len(states))
        self.assertEqual(vector_counts[-4:], [8, 1, 8, 1])
        self.assertEqual(result["codec_anchor_queries"], len(states))
        self.assertEqual(
            result["codec_anchor_bytes"], len(states) * (512 * 512 + 64 * 512) * 4
        )
        self.assertIsNone(owners[0]._bank)
        self.assertIsNone(owners[0]._temporary)
        self.assertEqual(list(self.root.glob("remnant-anchors-*")), [])
        rng = np.random.default_rng(42)
        expected = []
        for _ in range(2):
            order = rng.permutation(len(states)).tolist()
            expected.extend((tuple(order[start : start + 8]), True) for start in (0, 8))
        self.assertEqual(events[-4:], expected)

        # Failure after anchors exist must release their file and mappings too.
        with self.assertRaisesRegex(RuntimeError, "checkpoint failure"):
            fit(
                self.math,
                self.math.reference(),
                states[:2],
                mapped["development"],
                torch.zeros(512),
                mode="conditional_expanded",
                epochs=2,
                gradient_mode="anchored",
                scratch_root=self.root,
                checkpoint=Mock(side_effect=RuntimeError("checkpoint failure")),
            )
        self.assertEqual(list(self.root.glob("remnant-anchors-*")), [])
        with patch(
            "training.data.batches.shutil.disk_usage",
            return_value=SimpleNamespace(free=0),
        ):
            with self.assertRaisesRegex(ValueError, "warm anchor scratch.*requires"):
                fit(
                    self.math,
                    self.math.reference(),
                    states[:1],
                    mapped["development"],
                    torch.zeros(512),
                    mode="conditional_expanded",
                    epochs=1,
                    gradient_mode="anchored",
                    scratch_root=self.root,
                )
        self.assertEqual(list(self.root.glob("remnant-anchors-*")), [])

        with StateBatches(states[:2], "cpu", 2, self.root) as batches:
            # Bind simple independent offsets to test association, order and exact bytes.
            batches.prepare_anchors(
                lambda s: {
                    "state": s,
                    "rows": s["raw"] * 0.25,
                    "output": s["target"] * 0.5,
                }
            )
            with batches.stage([1, 0]) as batch:
                self.assertTrue(
                    torch.equal(batch.anchors[0]["rows"], states[1]["raw"] * 0.25)
                )
                self.assertTrue(
                    torch.equal(batch.anchors[1]["output"], states[0]["target"] * 0.5)
                )
            with batches.stage([1, 1]) as batch:
                self.assertEqual([s["ordinal"] for s in batch.states], [1, 1])
                self.assertTrue(
                    torch.equal(batch.anchors[0]["rows"], batch.anchors[1]["rows"])
                )
            with self.assertRaisesRegex(ValueError, "batch limit"):
                with batches.stage([0, 1, 0]):
                    self.fail("over-limit batch admitted")
            states[0]["ordinal"] = -1
            with self.assertRaisesRegex(ValueError, "association changed"):
                with batches.stage([0]):
                    self.fail("changed state admitted")
            states[0]["ordinal"] = 0
        with self.assertRaisesRegex(RuntimeError, "builder failure"):
            with StateBatches(states, "cpu", 2, self.root) as batches:
                batches.prepare_anchors(
                    Mock(side_effect=RuntimeError("builder failure"))
                )
        self.assertEqual(list(self.root.glob("remnant-anchors-*")), [])

        # A real loss failure retains its anchor argument in the traceback.
        # Such tensors must own their bytes before the mmap is closed.
        failure_traceback = None
        try:
            with StateBatches(states[:1], "cpu", 1, self.root) as batches:
                batches.prepare_anchors(lambda s: {
                    "state": s, "rows": s["raw"] * 0.25, "output": s["target"] * 0.5
                })
                bank_start = batches._bank.ctypes.data
                bank_end = bank_start + batches.anchor_bytes
                with batches.stage([0]) as batch:
                    losses.anchored_relaxed_forward_error(
                        self.math, batch.states[0],
                        Mock(side_effect=RuntimeError("anchor forward failure")),
                        0.02, batch.anchors[0],
                    )
        except RuntimeError as error:
            self.assertEqual(str(error), "anchor forward failure")
            failure_traceback = error.__traceback__
        self.assertIsNotNone(failure_traceback)
        while failure_traceback.tb_frame.f_code.co_name != "anchored_relaxed_forward_error":
            failure_traceback = failure_traceback.tb_next
        anchor = failure_traceback.tb_frame.f_locals["anchor"]
        for key in ("rows", "output"):
            self.assertFalse(bank_start <= anchor[key].data_ptr() < bank_end)
        # Do not dereference a closed mapping until the ownership check passes.
        self.assertTrue(torch.equal(anchor["rows"], states[0]["raw"] * 0.25))
        self.assertTrue(torch.equal(anchor["output"], states[0]["target"] * 0.5))
        self.assertEqual(list(self.root.glob("remnant-anchors-*")), [])

        # Allocation failure before yield must also release the borrowed view
        # from the retained stage frame, not just clear successfully staged data.
        tensor_to = torch.Tensor.to

        def fail_anchor_copy(tensor, *args, **kwargs):
            if kwargs.get("copy"):
                raise RuntimeError("anchor copy failed")
            return tensor_to(tensor, *args, **kwargs)

        failure_traceback = None
        try:
            with StateBatches(states[:1], "cpu", 1, self.root) as batches:
                batches.prepare_anchors(lambda s: {
                    "state": s, "rows": s["raw"] * 0.25, "output": s["target"] * 0.5
                })
                with patch.object(torch.Tensor, "to", fail_anchor_copy):
                    with batches.stage([0]):
                        self.fail("failed copy admitted a batch")
        except RuntimeError as error:
            self.assertEqual(str(error), "anchor copy failed")
            failure_traceback = error.__traceback__
        self.assertIsNotNone(failure_traceback)
        while failure_traceback.tb_frame.f_code.co_name != "stage":
            failure_traceback = failure_traceback.tb_next
        self.assertNotIn("values", failure_traceback.tb_frame.f_locals)
        self.assertEqual(list(self.root.glob("remnant-anchors-*")), [])

    # Capture orchestration: external programs are mocked, never launched.
    def test_capture_host_lifecycle_without_external_processes(self):
        model, scratch, published = (
            self.root / name for name in ("model", "scratch", "published")
        )
        model.mkdir()
        scratch.mkdir()
        self.math.atomic_json(
            scratch / "capture-config.json",
            {"storage_schema_version": 2, "audit_shared": True, "concurrency": 2},
        )
        (scratch / "replay.jsonl").write_text("")
        tools, temporary = self.root / "tools", self.root / "temporary"
        tools.mkdir()
        temporary.mkdir()
        log = self.root / "commands.jsonl"
        helper = tools / "fake-command"
        # Every external Docker/GPU/timeout command resolves to this test-owned
        # executable. It records intent; it never runs the embedded image command.
        helper.write_text(f"#!{sys.executable}\n" + r"""
import json, os, signal, sys
from pathlib import Path
name, args = Path(sys.argv[0]).name, sys.argv[1:]
with Path(os.environ["COMMAND_LOG"]).open("a") as stream:
    stream.write(json.dumps([name, *args]) + "\n")
scenario = os.environ["SCENARIO"]
if name == "timeout":
    while args and args[0].startswith("--"):
        args.pop(0)
    args.pop(0)
    if args[0] not in ("docker", "nvidia-smi"):
        raise SystemExit("unexpected external executable")
    os.execvpe(args[0], args, os.environ)
if name == "nvidia-smi":
    if scenario == "occupied":
        print("999")
elif args[:2] == ["image", "inspect"]:
    if scenario == "missing":
        sys.exit(1)
    print("sha256:" + "1"*64, "false" if scenario == "unlabeled" else "true")
elif args[0] == "run":
    gpu = "--gpus" in args
    cid = ("a" if gpu else "c") * 64
    Path(args[args.index("--cidfile")+1]).write_text(cid + "\n")
    if scenario == "preflight" and not gpu:
        sys.exit(1)
    if gpu:
        if scenario == "timeout": sys.exit(124)
        if scenario == "publication": sys.exit(1)
        if scenario == "interrupt":
            os.kill(os.getppid(), signal.SIGTERM)
            sys.exit(143)
elif args[0] == "stop":
    if scenario == "stop-failure" and args[-1] == "a"*64:
        sys.stderr.write("daemon unavailable")
        sys.exit(1)
else:
    raise SystemExit("unexpected Docker operation")
""")
        helper.chmod(0o755)
        for command in ("docker", "nvidia-smi", "timeout"):
            (tools / command).symlink_to(helper.name)
        launcher = (
            Path(__file__).resolve().parents[2] / "scripts/local/capture-latent.sh"
        )
        arguments = [
            "bash",
            str(launcher),
            "run",
            "prepared:tag",
            str(model),
            str(scratch),
            str(published),
            "100",
        ]
        environment = {
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.defpath,
            "COMMAND_LOG": str(log),
            "TMPDIR": str(temporary),
            "CAPTURE_GPUS": "4,5,6,7",
        }
        for scenario in (
            "success",
            "missing",
            "unlabeled",
            "preflight",
            "occupied",
            "publication",
            "timeout",
            "interrupt",
            "stop-failure",
        ):
            with self.subTest(scenario=scenario):
                log.unlink(missing_ok=True)
                outcome = subprocess.run(
                    arguments,
                    env={**environment, "SCENARIO": scenario},
                    text=True,
                    capture_output=True,
                    timeout=15,
                )
                self.assertEqual(
                    outcome.returncode == 0, scenario == "success", outcome.stderr
                )
                if scenario == "timeout":
                    self.assertEqual(outcome.returncode, 124)
                if scenario == "interrupt":
                    self.assertEqual(outcome.returncode, 143)
                commands = [json.loads(line) for line in log.read_text().splitlines()]
                docker = [args[1:] for args in commands if args[0] == "docker"]
                runs = [args for args in docker if args[0] == "run"]
                gpu = [args for args in runs if "--gpus" in args]
                stops = [args[-1] for args in docker if args[0] == "stop"]
                self.assertFalse(
                    any(args[0] in ("build", "pull", "exec") for args in docker)
                )
                if scenario in ("missing", "unlabeled"):
                    self.assertEqual(runs, [])
                    self.assertEqual(stops, [])
                    continue
                cpu = runs[0]
                self.assertNotIn("--gpus", cpu)
                self.assertIn("_preflight", cpu)
                self.assertIn("/capture-repo/scripts/local/capture-latent.sh", cpu)
                if scenario in ("preflight", "occupied"):
                    self.assertEqual(gpu, [])
                    self.assertEqual(stops, ["c" * 64])
                else:
                    self.assertEqual(stops, ["c" * 64, "a" * 64])
                    self.assertEqual(
                        gpu[0][gpu[0].index("--gpus") + 1], '"device=4,5,6,7"'
                    )
                    self.assertIn("_capture", gpu[0])
                    self.assertTrue(gpu[0][-2].isdigit())  # Absolute startup/replay deadline.
                    self.assertIn("sha256:" + "1" * 64, gpu[0])
                for args in runs:
                    self.assertEqual(args[args.index("--pull") + 1], "never")
                    self.assertIn(
                        "type=bind,src="
                        + str(model.resolve())
                        + ",dst=/model,readonly",
                        args,
                    )
                    if scenario != "stop-failure":
                        self.assertFalse(
                            Path(args[args.index("--cidfile") + 1]).exists()
                        )
        log.unlink()
        for arguments in (["--help"], ["build", "production", "capture"]):
            result = subprocess.run(
                ["bash", str(launcher), *arguments],
                env=environment,
                text=True,
                capture_output=True,
                timeout=5,
            )
            self.assertEqual(result.returncode == 0, arguments == ["--help"])
            self.assertFalse(log.exists())

    def test_capture_preflight_and_server_lifecycle(self):
        import socket

        model, published = self.root / "model", self.root / "published"
        model.mkdir()
        (self.root / "replay.jsonl").write_text("")
        config = {
            "storage_schema_version": 2,
            "audit_shared": True,
            "concurrency": 2,
            "layers": [2],
            "input_manifest": {"model_revision": "test"},
        }
        config_path = self.root / "capture-config.json"
        self.math.atomic_json(config_path, config)
        preflight = self.math.capture_preflight
        with patch.object(
            self.math.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=100 * 1024**3),
        ), patch.object(socket, "socket"):
            with patch.object(
                sys,
                "argv",
                [
                    str(self.runner),
                    "capture-preflight",
                    "--root",
                    str(self.root),
                    "--publish-root",
                    str(published),
                    "--model",
                    str(model),
                ],
            ):
                self.math.main()
            config["audit_shared"] = False
            self.math.atomic_json(config_path, config)
            proof = {
                "status": "pass",
                "kind": "audited-live-shared-capture",
                "issues": 2,
                "layers": [2],
                "model_revision": "test",
                "collector_sha256": file_sha256(
                    self.runner.parents[2] / "srt/remnant/capture.py"
                ),
                "runner_sha256": file_sha256(self.runner),
            }
            self.math.atomic_json(self.root / "shared-audit-proof.json", proof)
            preflight(self.root, published, model, 30312)
            proof["kind"] = "conversion-only"
            self.math.atomic_json(self.root / "shared-audit-proof.json", proof)
            with self.assertRaisesRegex(ValueError, "audited live-capture proof"):
                preflight(self.root, published, model, 30312)
            config["audit_shared"], config["concurrency"] = True, 3
            self.math.atomic_json(config_path, config)
            with self.assertRaisesRegex(ValueError, "concurrency two"):
                preflight(self.root, published, model, 30312)
            config["concurrency"] = 2
            self.math.atomic_json(config_path, config)
            with patch.object(
                self.math.shutil, "disk_usage", return_value=SimpleNamespace(free=0)
            ):
                with self.assertRaisesRegex(ValueError, "50 GiB"):
                    preflight(self.root, published, model, 30312)
            with patch(
                "tempfile.TemporaryFile", side_effect=OSError("Read-only file system")
            ):
                with self.assertRaisesRegex(OSError, "Read-only"):
                    preflight(self.root, published, model, 30312)
            with self.assertRaisesRegex(ValueError, "destinations must differ"):
                preflight(self.root, self.root, model, 30312)

        # Exercise the actual shell lifecycle with fake external tools. setsid
        # creates real, test-owned groups, but never executes the model launcher.
        import time

        tools = self.root / "lifecycle-tools"
        tools.mkdir()
        log = self.root / "lifecycle.jsonl"
        helper = tools / "fake-command"
        helper.write_text(f"#!{sys.executable}\n" + r"""
import json, os, signal, sys, time
from pathlib import Path
name, args = Path(sys.argv[0]).name, sys.argv[1:]
scenario = os.environ["SCENARIO"]
with Path(os.environ["COMMAND_LOG"]).open("a") as stream:
    stream.write(json.dumps({"tool": name, "args": args, "pid": os.getpid(),
                            "environment": {k: os.environ.get(k) for k in (
                                "TP", "MAX_RUN", "SERVER_SEED", "DECODE_CFG",
                                "NSYS_OUTPUT", "REQUEST_METRICS_DIR", "STORE_IMPLEMENTATION")}}) + "\n")
if name == "setsid":
    os.setsid()
    if args[0] == "bash":
        if scenario == "startup":
            sys.exit(7)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        while True:
            time.sleep(.1)
    os.execvpe(args[0], args, os.environ)
elif name == "timeout":
    while args[0].startswith("--"):
        args.pop(0)
    args.pop(0)
    if args[0] != "python3":
        raise SystemExit("unexpected timeout command")
    os.execvpe(args[0], args, os.environ)
elif name == "python3":
    if "capture-preflight" in args:
        sys.exit(1 if scenario == "preflight" else 0)
    if "capture-client" not in args:
        raise SystemExit("unexpected Python invocation")
    if scenario == "publication":
        sys.exit(1)
    if scenario == "client-timeout":
        sys.exit(124)
    if scenario in ("interrupt", "repeat-interrupt"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.kill(os.getppid(), signal.SIGINT)
        if scenario == "repeat-interrupt":
            time.sleep(.05)
            os.kill(os.getppid(), signal.SIGTERM)
        sys.exit(130)
elif name == "curl":
    if scenario in ("timeout", "startup"):
        sys.exit(7)
    time.sleep(1.1)  # Startup must reduce the client's remaining budget.
elif name == "sleep":
    time.sleep(min(.1, float(args[0])))
else:
    raise SystemExit("unexpected external tool")
""")
        helper.chmod(0o755)
        for tool in ("setsid", "timeout", "python3", "curl", "sleep"):
            (tools / tool).symlink_to(helper.name)
        launcher = (
            Path(__file__).resolve().parents[2] / "scripts/local/capture-latent.sh"
        )
        environment = {
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.defpath,
            "COMMAND_LOG": str(log),
            "NSYS_OUTPUT": "must-not-profile",
            "REQUEST_METRICS_DIR": "must-not-record",
            "STORE_IMPLEMENTATION": "must-not-pack",
        }
        for scenario in (
            "success",
            "preflight",
            "startup",
            "timeout",
            "client-timeout",
            "interrupt",
            "repeat-interrupt",
            "publication",
            "expired",
        ):
            with self.subTest(scenario=scenario):
                log.unlink(missing_ok=True)
                # Let preflight launch the fake server before its unhealthy
                # health checks exhaust the budget; one second can expire first.
                deadline = int(time.time()) + 6
                if scenario == "expired":
                    deadline = 0
                try:
                    result = subprocess.run(
                        [
                            "bash",
                            str(launcher),
                            "_capture",
                            str(self.root),
                            str(published),
                            str(model),
                            str(self.runner),
                            str(deadline),
                            "30312",
                        ],
                        env={**environment, "SCENARIO": scenario},
                        capture_output=True,
                        text=True,
                        timeout=12,
                    )
                    self.assertEqual(
                        result.returncode == 0, scenario == "success", result.stderr
                    )
                    events = (
                        [json.loads(line) for line in log.read_text().splitlines()]
                        if log.exists()
                        else []
                    )
                    groups = [e for e in events if e["tool"] == "setsid"]
                    for event in groups:
                        with self.assertRaises(ProcessLookupError):
                            os.killpg(event["pid"], 0)
                    servers = [e for e in groups if e["args"][0] == "bash"]
                    clients = [
                        e
                        for e in events
                        if e["tool"] == "python3" and "capture-client" in e["args"]
                    ]
                    self.assertEqual(
                        bool(servers), scenario not in ("preflight", "expired")
                    )
                    self.assertEqual(
                        bool(clients),
                        scenario
                        in (
                            "success",
                            "publication",
                            "client-timeout",
                            "interrupt",
                            "repeat-interrupt",
                        ),
                    )
                    if scenario == "client-timeout":
                        self.assertEqual(result.returncode, 124)
                    if clients:
                        args, env = clients[0]["args"], clients[0]["environment"]
                        self.assertLess(int(args[args.index("--seconds") + 1]), 6)
                        self.assertEqual(
                            args[args.index("--publish-root") + 1], str(published)
                        )
                        self.assertEqual(env["TP"], "4")
                        self.assertEqual(env["MAX_RUN"], "2")
                        self.assertEqual(env["SERVER_SEED"], "42")
                        self.assertEqual(
                            json.loads(env["DECODE_CFG"])["decode"]["backend"],
                            "disabled",
                        )
                        for name in (
                            "NSYS_OUTPUT",
                            "REQUEST_METRICS_DIR",
                            "STORE_IMPLEMENTATION",
                        ):
                            self.assertIsNone(env[name])
                    if servers:
                        self.assertIn("native", servers[0]["args"])
                        self.assertTrue((self.root / "server.log").exists())
                finally:
                    # A regression must not leave test-owned workers behind.
                    for line in log.read_text().splitlines() if log.exists() else []:
                        event = json.loads(line)
                        if event["tool"] == "setsid":
                            try:
                                os.killpg(event["pid"], signal.SIGKILL)
                            except ProcessLookupError:
                                pass

    # Exact evaluation, reports, and publication failure handling.
    def test_comparison_masks_counts_groups_and_ties(self):
        raw = torch.stack(
            (torch.arange(512).float(), -torch.arange(512).float(), torch.zeros(512))
        )
        masks = evaluate.comparison_masks(raw, raw.abs())
        ref = self.math.reference()
        for method in evaluate.MASK_METHODS:
            mask = masks[method.key]
            self.assertEqual(mask.dtype, torch.bool)
            self.assertTrue(bool((mask.sum(-1) == method.retained).all()))
            if method.key.startswith("structured"):
                size, kept = (4, 2) if method.retained == 256 else (8, 3)
                self.assertTrue(bool((mask.reshape(3, -1, size).sum(-1) == kept).all()))
                expected = torch.arange(size) >= size - kept
                self.assertTrue(
                    torch.equal(
                        mask[0].reshape(-1, size), expected.expand(512 // size, -1)
                    )
                )
            else:
                self.assertTrue(
                    torch.equal(mask, ref.topmag_keep_mask(raw, method.retained / 512))
                )
        for scores, retained, size in (
            (raw, 193, 8),
            (raw, 192, 4),
            (raw.double(), 256, 4),
            (torch.full_like(raw, float("nan")), 256, 512),
        ):
            with self.assertRaisesRegex(ValueError, "supported mask budget"):
                evaluate.ranking_mask(scores, retained, group_size=size)

    def test_comparison_codec_parity_and_independent_192_output(self):
        state = self._state("development", 0)
        ref = self.math.reference()
        scorer = ExpandedConditionalScore(torch.zeros(512), seed=50)
        masks = evaluate.comparison_masks(
            state["raw"], scorer(state["raw"], state["eps"])
        )
        with torch.no_grad():
            for method in evaluate.MASK_METHODS[:3]:
                mask = masks[method.key]
                args = (
                    self.math,
                    ref,
                    state["raw"],
                    mask,
                    state["weight"],
                    state["eps"],
                    state["freq"],
                )
                self.assertTrue(
                    torch.equal(packed_rows(*args), simulated_packed_rows(*args))
                )
            self.assertEqual(
                float(forward_error(self.math, ref, state, scorer)),
                float(
                    forward_error(
                        self.math, ref, state, scorer, keep_mask=masks["learned_256"]
                    )
                ),
            )
        # Independently derive constant per-tile BF16 values/scales with Python
        # scalars, rather than using the quantization helper under test.
        state["raw"] = torch.ones_like(state["raw"])
        state["weight"] = torch.tensor(
            [2.0**tile for tile in range(8)]
        ).repeat_interleave(64)
        mask = (torch.arange(512) % 8 >= 5).expand(512, -1)
        expected = torch.zeros_like(state["raw"])
        for tile in range(8):
            normalized = float(
                torch.tensor(
                    (2.0**tile) / math.sqrt(192 / 512 + state["eps"])
                ).bfloat16()
            )
            scale = 2.0 ** math.ceil(math.log2(max(abs(normalized), 1e-4) / 448))
            code = torch.tensor(normalized / scale).to(torch.float8_e4m3fn).float()
            expected[:, tile * 64 : (tile + 1) * 64] = (code * scale).bfloat16().float()
        expected.masked_fill_(~mask, 0)
        expected = self.math.rotate(expected, state["freq"]).bfloat16().float()
        args = (
            self.math,
            ref,
            state["raw"],
            mask,
            state["weight"],
            state["eps"],
            state["freq"],
        )
        self.assertTrue(
            torch.equal(packed_rows(*args, retained_features=192), expected)
        )
        # Independent softmax with the sink as a zero-valued extra KV row.
        kv = torch.cat((state["swa"], expected))
        logits = torch.cat(
            (state["q"] @ kv.T * state["scale"], state["sink"][:, None]), -1
        )
        probability = (logits - logits.amax(-1, keepdim=True)).exp()
        probability = probability / probability.sum(-1, keepdim=True)
        output = rounded_output(
            self.math,
            probability @ torch.cat((kv, torch.zeros(1, 512))),
            state["query_freq"],
        )
        expected_error = (
            (output - state["target"]).square().sum().double()
            / state["reference_squared_norm"]
        ).sqrt()
        actual = forward_error(
            self.math, ref, state, scorer, keep_mask=mask, retained_features=192
        )
        torch.testing.assert_close(actual, expected_error, atol=2e-6, rtol=2e-5)
        for budget, explicit in ((192, None), (256, mask)):
            with self.assertRaisesRegex(ValueError, "inference-only"):
                forward_error(
                    self.math,
                    ref,
                    state,
                    scorer,
                    0.02,
                    retained_features=budget,
                    keep_mask=explicit,
                )
        with self.assertRaisesRegex(ValueError, "256 survivors"):
            packed_rows(*args)

    def test_comparison_weighting_p95_and_paired_intervals(self):
        # Issue a/layer 2 has three queries; every other issue/layer has one.
        # A pooled mean would be wrong: the intended issue means are 3 and 7.
        base = [
            {"rid": "a", "layer": 2, "ordinal": q, "error": 1.0} for q in range(3)
        ] + [
            {"rid": "a", "layer": 4, "ordinal": 0, "error": 5.0},
            {"rid": "b", "layer": 2, "ordinal": 0, "error": 3.0},
            {"rid": "b", "layer": 4, "ordinal": 0, "error": 11.0},
        ]
        records = {
            m.key: [
                {**r, "error": r["error"] + (1 if m.key.startswith("learned") else 0)}
                for r in base
            ]
            for m in evaluate.MASK_METHODS
        }
        identities = {(r["rid"], r["layer"], r["ordinal"]) for r in base}
        summary = evaluate.summarize_comparison(records, identities)
        self.assertEqual(summary, evaluate.summarize_comparison(records, identities))
        self.assertEqual(summary[0]["mean"], 5.0)
        self.assertEqual(summary[0]["issue_means"], {"a": 3.0, "b": 7.0})
        self.assertEqual(summary[0]["p95"], 9.5)
        self.assertEqual(summary[0]["mean_ci95"], [3.0, 7.0])
        self.assertEqual(
            summary[2]["paired_vs_magnitude"]["mean_difference_ci95"], [1.0, 1.0]
        )
        for faulty in (
            records["topmag_256"][:-1],
            records["topmag_256"] + [base[0]],
            [{**r, "ordinal": r["ordinal"] + 99} for r in base],
            [{**r, "error": float("nan")} for r in base],
        ):
            with self.assertRaisesRegex(ValueError, "comparison records"):
                evaluate.summarize_comparison(
                    {**records, "topmag_256": faulty}, identities
                )

    def test_retention_sweep_budgets_and_fixed_eight_coordinate_groups(self):
        raw = torch.stack((torch.arange(512).float(), torch.zeros(512)))
        masks = evaluate.comparison_masks(
            raw, raw.abs(), methods=evaluate.SWEEP_METHODS
        )
        self.assertEqual(len(masks), 21)
        for method in evaluate.SWEEP_METHODS:
            mask = masks[method.key]
            self.assertTrue(bool((mask.sum(-1) == method.retained).all()))
            if method.key.startswith("structured"):
                self.assertEqual(method.group_size, 8)
                self.assertTrue(
                    bool(
                        (mask.reshape(2, 64, 8).sum(-1) == method.retained // 64).all()
                    )
                )
                expected = torch.arange(8) >= 8 - method.retained // 64
                self.assertTrue(
                    torch.equal(mask[0].reshape(64, 8), expected.expand(64, -1))
                )
        # 4:8 is deliberately less constrained than 2:4: this row separates
        # the two patterns instead of silently relabeling the old 2:4 result.
        skewed = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0]).repeat(
            1, 64
        )
        eight = evaluate.ranking_mask(skewed, 256, group_size=8)
        four = evaluate.ranking_mask(skewed, 256, group_size=4)
        self.assertFalse(torch.equal(eight, four))
        self.assertTrue(bool((eight.reshape(1, 64, 8)[..., 4:]).all()))
        state = self._state("development", 0)
        for method in evaluate.SWEEP_METHODS:
            if not method.key.startswith("structured"):
                continue
            mask = evaluate.ranking_mask(
                state["raw"].abs(), method.retained, group_size=8
            )
            decoded = packed_rows(
                self.math,
                self.math.reference(),
                state["raw"],
                mask,
                state["weight"],
                state["eps"],
                state["freq"],
                retained_features=method.retained,
            )
            self.assertEqual(decoded.shape, state["raw"].shape)
            self.assertTrue(bool(torch.isfinite(decoded).all()))
            # Key RoPE touches only the tail; omitted NoPE coordinates stay zero.
            self.assertTrue(bool((decoded[:, :448][~mask[:, :448]] == 0).all()))

    def test_retention_sweep_cli_exports_two_three_line_figures(self):
        cache = self._cache(8)
        policy = self._warm_policy(8)
        output = self.root / "sweep.json"
        figures = self.root / "figures"
        args = [
            "--cache",
            str(cache),
            "--policy",
            str(policy),
            "--runner",
            str(self.runner),
            "--output",
            str(output),
            "--layers",
            "8",
            "--threads",
            "2",
            "--retention-sweep",
            "--figure-dir",
            str(figures),
        ]
        # Inspect the real artists and real PNG output, not mocked plotting.
        import matplotlib.pyplot as plt

        with patch.object(
            plt.Axes, "plot", autospec=True, side_effect=plt.Axes.plot
        ) as plot:
            result = evaluate.main(args)
        self.assertEqual(plot.call_count, 6)
        for call in plot.call_args_list:
            self.assertEqual(tuple(call.args[1]), evaluate.RETENTION_COUNTS)
            self.assertEqual(len(call.args[2]), 7)
        self.assertEqual(result["status"], "partial-layer diagnostic")
        self.assertEqual(
            result["retained_features"], [64, 128, 192, 256, 320, 384, 448]
        )
        self.assertEqual(len(result["summary"]), 21)
        self.assertEqual(result["coverage"]["methods"], 21)
        self.assertEqual(len(result["figures"]["files"]), 2)
        for path in result["figures"]["files"]:
            self.assertTrue(Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        patterned = {
            r["retained_features"]: r["structured_pattern"]
            for r in result["summary"]
            if r["method"].startswith("structured")
        }
        self.assertEqual(patterned[256], "4:8")
        self.assertEqual(patterned[64], "1:8")
        self.assertEqual(patterned[448], "7:8")
        self.assertIn("256-feature point is 4:8", output.with_suffix(".md").read_text())
        with output.with_suffix(".csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 21)
        self.assertTrue(
            all(
                float(a["mean"]) == b["mean"] and float(a["p95"]) == b["p95"]
                for a, b in zip(rows, result["summary"])
            )
        )
        # A failure on the second figure rolls back the first figure and tables.
        failed_output = self.root / "failed-figures.json"
        failed_figures = self.root / "failed-figures"
        link = os.link

        def fail_second_figure(source, target):
            if Path(target) == failed_figures / "attention-error-p95.png":
                raise OSError("second figure publication failed")
            return link(source, target)

        with patch("training.evaluate.os.link", side_effect=fail_second_figure):
            with self.assertRaisesRegex(OSError, "second figure publication failed"):
                evaluate._publish_comparison(
                    self.math, failed_output, dict(result), failed_figures
                )
        self.assertEqual(list(failed_figures.iterdir()), [])
        self.assertTrue(
            all(
                not failed_output.with_suffix(s).exists()
                for s in (".json", ".csv", ".md")
            )
        )
        # Existing figures must fail before any replay, even with a fresh JSON.
        args[args.index("--output") + 1] = str(self.root / "duplicate-figures.json")
        with patch(
            "training.evaluate.load_cached_layer",
            side_effect=AssertionError("must not replay"),
        ):
            with self.assertRaisesRegex(ValueError, "fresh"):
                evaluate.main(args)

    def test_six_method_cli_reports_and_default_evaluation_unchanged(self):
        cache = self._cache(8)
        policy = self._warm_policy(8)
        output = self.root / "comparison.json"
        args = [
            "--cache",
            str(cache),
            "--policy",
            str(policy),
            "--runner",
            str(self.runner),
            "--output",
            str(output),
            "--layers",
            "8",
            "--threads",
            "2",
            "--compare-masks",
        ]
        result = evaluate.main(args)
        self.assertEqual(result["status"], "partial-layer diagnostic")
        self.assertEqual(result["coverage"]["query_layer_examples"], 10)
        self.assertEqual(result["coverage"]["codec_256_equivalence_checks"], 30)
        self.assertFalse(result["fresh_test_completed"])
        self._assert_run_settings(result["settings"], cache, policy)
        self.assertEqual(json.loads(output.read_text()), result)
        identities = [
            {(r["rid"], r["layer"], r["ordinal"]) for r in rows}
            for rows in result["records"].values()
        ]
        self.assertTrue(all(keys == identities[0] for keys in identities))
        with output.with_suffix(".csv").open() as stream:
            flat = list(csv.DictReader(stream))
        self.assertEqual(len(flat), 6)
        text = output.with_suffix(".md").read_text()
        self.assertIn("partial-layer diagnostic", text)
        self.assertIn("not retrained at 192", text)
        for row, serialized in zip(result["summary"], flat):
            self.assertEqual(float(serialized["mean"]), row["mean"])
            self.assertEqual(float(serialized["p95"]), row["p95"])
            self.assertIn(f"{100*row['mean']:.4f}% | {100*row['p95']:.4f}%", text)
        # Old CLI schema and bit-exact 256-value results stay intact.
        plain_output = self.root / "ordinary.json"
        plain = evaluate.main(
            [
                a if a != str(output) else str(plain_output)
                for a in args
                if a != "--compare-masks"
            ]
        )
        self._assert_run_settings(plain["settings"], cache, policy)
        dev = plain["layers"][0]["roles"]["development"]
        by_method = {r["method"]: r for r in result["summary"]}
        self.assertEqual(dev["selected"]["mean"], by_method["learned_256"]["mean"])
        self.assertEqual(dev["topmag"]["p95"], by_method["topmag_256"]["p95"])
        previous = {
            suffix: output.with_suffix(suffix).read_bytes()
            for suffix in (".json", ".csv", ".md")
        }
        with self.assertRaisesRegex(SystemExit, "2"):
            evaluate.main(args)
        self.assertTrue(
            all(output.with_suffix(s).read_bytes() == b for s, b in previous.items())
        )
        # Sidecars also block publication even if no JSON exists.
        occupied = self.root / "occupied.json"
        occupied.with_suffix(".csv").write_text("keep")
        with self.assertRaisesRegex(ValueError, "fresh"):
            evaluate.main([str(occupied) if a == str(output) else a for a in args])
        self.assertEqual(occupied.with_suffix(".csv").read_text(), "keep")

    def test_corrupt_cache_payload_and_receipt_are_still_rejected(self):
        cache = self._cache(8)
        for suffix in (".json", ".pt"):
            path = cache / f"layer-08{suffix}"
            original_size = path.stat().st_size
            try:
                with path.open("ab") as stream:
                    stream.write(b"\\n")
                with self.subTest(suffix=suffix), self.assertRaisesRegex(
                    ValueError, "cached (payload checksum|layer receipt) changed"
                ):
                    load_cached_layer(cache, 8)
            finally:
                with path.open("r+b") as stream:
                    stream.truncate(original_size)

    def test_report_publication_rolls_back_only_owned_artifacts(self):
        cache = self._cache(8)
        policy = self._warm_policy(8)
        arguments = [
            "--compare-masks",
            "--cache",
            str(cache),
            "--policy",
            str(policy),
            "--runner",
            str(self.runner),
            "--layers",
            "8",
            "--threads",
            "2",
        ]
        result = evaluate.main(
            [*arguments, "--output", str(self.root / "reference.json")]
        )
        link = os.link
        temporary_directory = tempfile.TemporaryDirectory

        @contextmanager
        def cleanup_failure(*args, **kwargs):
            with temporary_directory(*args, **kwargs) as scratch:
                yield scratch
            raise OSError("evaluation scratch cleanup failed")

        output = self.root / "comparison-cleanup.json"
        with patch("training.evaluate.tempfile.TemporaryDirectory", side_effect=cleanup_failure):
            with self.assertRaisesRegex(OSError, "scratch cleanup failed"):
                evaluate._publish_comparison(self.math, output, dict(result), None)
        self.assertTrue(all(not output.with_suffix(s).exists() for s in (".json", ".csv", ".md")))

        # Ordinary evaluation must have the same interruption/cleanup ownership
        # guarantees as the multi-file comparison publisher.
        for failure in ("interrupt", "competitor", "hard-link", "cleanup"):
            output = self.root / f"ordinary-{failure}.json"
            competing_bytes = []

            def ordinary_link(source, target):
                if failure == "competitor":
                    Path(target).write_text("another writer owns this")
                elif failure == "hard-link":
                    competing_bytes.append(Path(source).read_bytes())
                    link(source, target)
                link(source, target)
                if failure == "interrupt":
                    raise KeyboardInterrupt()

            with self.subTest(ordinary_publication=failure), patch(
                "training.evaluate.os.link", side_effect=ordinary_link
            ), patch(
                "training.evaluate.tempfile.TemporaryDirectory",
                side_effect=cleanup_failure if failure == "cleanup" else temporary_directory,
            ):
                with self.assertRaises(KeyboardInterrupt if failure == "interrupt" else OSError):
                    evaluate.main([*arguments[1:], "--output", str(output)])
            if failure == "competitor":
                self.assertEqual(output.read_text(), "another writer owns this")
            elif failure == "hard-link":
                self.assertEqual(output.read_bytes(), competing_bytes[0])
            else:
                self.assertFalse(output.exists())

        # A progress path acquired after freshness preflight is still unowned.
        output = self.root / "competing-progress.json"
        partial = output.with_suffix(".partial.json")
        load = evaluate.load_cached_layer

        def competing_progress(*args, **kwargs):
            states = load(*args, **kwargs)
            partial.write_text("another writer owns this")
            return states

        with patch("training.evaluate.load_cached_layer", side_effect=competing_progress):
            with self.assertRaises(FileExistsError):
                evaluate.main([*arguments, "--output", str(output)])
        self.assertEqual(partial.read_text(), "another writer owns this")
        self.assertFalse(output.exists())

        # A replacement between an ownership check and the next update must
        # survive too; a check followed by rename is not an atomic claim.
        partial = self.root / "replaced-progress.partial.json"
        progress = evaluate._PartialReport(partial)
        progress.write(self.math, {"status": "partial"})
        path_lstat = Path.lstat
        replacements = []

        def replaced_after_stat(path, *args, **kwargs):
            identity = path_lstat(path, *args, **kwargs)
            if path == partial and not replacements:
                other = self.root / "other-progress.json"
                other.write_text("replacement belongs to another writer")
                other.replace(partial)
                replacements.append(partial)
            return identity

        with patch.object(Path, "lstat", replaced_after_stat):
            progress.write(self.math, {"status": "failed"})
        self.assertEqual(replacements, [partial])
        self.assertEqual(partial.read_text(), "replacement belongs to another writer")

        # Also replace the name after the opened descriptor is verified but
        # before writing; the write must stay on our old, now-unlinked inode.
        partial = self.root / "replaced-after-open.partial.json"
        progress = evaluate._PartialReport(partial)
        progress.write(self.math, {"status": "partial"})
        file_stat = os.fstat
        replacements.clear()

        def replaced_after_open(descriptor):
            identity = file_stat(descriptor)
            if not replacements:
                other = self.root / "other-open-progress.json"
                other.write_text("replacement after descriptor verification")
                other.replace(partial)
                replacements.append(partial)
            return identity

        with patch("training.evaluate.os.fstat", side_effect=replaced_after_open):
            with self.assertRaisesRegex(FileExistsError, "ownership lost"):
                progress.write(self.math, {"status": "failed"})
        self.assertEqual(replacements, [partial])
        self.assertEqual(partial.read_text(), "replacement after descriptor verification")
        for suffix in (".csv", ".md", ".json"):
            output = self.root / f"failed-{suffix[1:]}.json"

            def interrupted(source, target):
                if Path(target) == output.with_suffix(suffix):
                    raise OSError("injected publication failure")
                return link(source, target)

            with self.subTest(suffix=suffix), patch(
                "training.evaluate.os.link", side_effect=interrupted
            ):
                with self.assertRaisesRegex(OSError, "injected publication failure"):
                    evaluate._publish_comparison(self.math, output, dict(result), None)
            self.assertTrue(
                all(
                    not output.with_suffix(s).exists() for s in (".json", ".csv", ".md")
                )
            )

        output = self.root / "competing.json"

        def competitor(source, target):
            if Path(target) == output.with_suffix(".md"):
                Path(target).write_text("another writer owns this")
            return link(source, target)

        with patch("training.evaluate.os.link", side_effect=competitor):
            with self.assertRaises(FileExistsError):
                evaluate._publish_comparison(self.math, output, dict(result), None)
        self.assertEqual(
            output.with_suffix(".md").read_text(), "another writer owns this"
        )
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".csv").exists())

        output = self.root / "competing-link.json"

        def competing_symlink(source, target):
            if Path(target) == output.with_suffix(".md"):
                Path(target).symlink_to(source)
            return link(source, target)

        with patch("training.evaluate.os.link", side_effect=competing_symlink):
            with self.assertRaises(FileExistsError):
                evaluate._publish_comparison(self.math, output, dict(result), None)
        self.assertTrue(output.with_suffix(".md").is_symlink())
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".csv").exists())

        output = self.root / "competing-hard-link.json"

        def competing_hard_link(source, target):
            if Path(target) == output.with_suffix(".md"):
                link(source, target)
            return link(source, target)

        with patch("training.evaluate.os.link", side_effect=competing_hard_link):
            with self.assertRaises(FileExistsError):
                evaluate._publish_comparison(self.math, output, dict(result), None)
        self.assertTrue(output.with_suffix(".md").is_file())
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".csv").exists())

        # A signal immediately after creating the marker must still roll back
        # by inode ownership, even though the publication call never returned.
        output = self.root / "interrupt.json"

        def interrupted_after_link(source, target):
            link(source, target)
            if Path(target) == output:
                raise KeyboardInterrupt()

        with patch("training.evaluate.os.link", side_effect=interrupted_after_link):
            with self.assertRaises(KeyboardInterrupt):
                evaluate._publish_comparison(self.math, output, dict(result), None)
        self.assertTrue(
            all(not output.with_suffix(s).exists() for s in (".json", ".csv", ".md"))
        )
        self.assertEqual(list(self.root.glob(".evaluation-*")), [])

        # The ordinary evaluator must also use exclusive publication.
        output = self.root / "ordinary-race.json"

        def ordinary_competitor(source, target):
            if Path(target) == output:
                output.write_text("a different writer won")
            return link(source, target)

        with patch("training.evaluate.os.link", side_effect=ordinary_competitor):
            with self.assertRaises(FileExistsError):
                evaluate.main(
                    [a for a in arguments if a != "--compare-masks"]
                    + ["--output", str(output)]
                )
        self.assertEqual(output.read_text(), "a different writer won")

    def test_comparison_fails_closed_and_retains_partial_layers(self):
        cache = self._cache(8)
        policy = self._warm_policy(8)
        output = self.root / "failure.json"
        args = [
            "--cache",
            str(cache),
            "--policy",
            str(policy),
            "--runner",
            str(self.runner),
            "--output",
            str(output),
            "--layers",
            "8",
            "--threads",
            "2",
            "--compare-masks",
        ]
        with patch(
            "training.evaluate.summarize_comparison",
            side_effect=ValueError("coverage failed"),
        ):
            with self.assertRaisesRegex(ValueError, "coverage failed"):
                evaluate.main(args)
        partial = json.loads(output.with_suffix(".partial.json").read_text())
        self.assertEqual(partial["status"], "failed")
        self.assertEqual(partial["completed_layers"], [8])
        self.assertEqual(len(partial["records"]["learned_192"]), 10)
        self.assertTrue(
            all(not output.with_suffix(s).exists() for s in (".json", ".md", ".csv"))
        )
        # Missing checkpoints fail before measurement.
        args[args.index("--layers") + 1] = "8"
        args[args.index("--output") + 1] = str(self.root / "missing.json")
        (policy.parent / "warm-layer-08.json").unlink()
        with self.assertRaisesRegex(ValueError, "cannot read policy checkpoint"):
            evaluate.main(args)
        self.assertFalse((self.root / "missing.partial.json").exists())
        # Preserve the checkpoint for the remaining reader-boundary checks.
        scorer = ExpandedConditionalScore(torch.zeros(512), seed=50)
        checkpoint = policy.parent / "restored-layer-08.json"
        self.math.atomic_json(
            checkpoint,
            {
                "layer": 8,
                "selected_parameters": {
                    name: value.tolist() for name, value in scorer.state_dict().items()
                },
            },
        )
        manifest = json.loads(policy.read_text())
        manifest["selected"][0].update(
            path=checkpoint.name, sha256=file_sha256(checkpoint)
        )
        self.math.atomic_json(policy, manifest)
        receipt = cache / "layer-08.json"
        original = receipt.read_bytes()
        receipt.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "cached layer receipt changed"):
            evaluate.main(args)
        # Reader corruption is retained as a failed partial report.
        self.assertEqual(json.loads((self.root / "missing.partial.json").read_text())["status"], "failed")
        receipt.write_bytes(original)
        args[args.index("--output") + 1] = str(self.root / "denominator.json")
        states = evaluate.load_cached_layer(cache, 8)
        states["development"][0]["reference_squared_norm"] = 0
        with patch("training.evaluate.load_cached_layer", return_value=states):
            with self.assertRaisesRegex(ValueError, "positive native denominator"):
                evaluate.main(args)
        self.assertTrue(
            all(
                not (self.root / "denominator.json").with_suffix(s).exists()
                for s in (".json", ".csv", ".md")
            )
        )


class TrainingCudaTests(unittest.TestCase):
    """Opt-in SM90 execution of the existing cache, scorer and loss workflows.

    REMNANT_RUN_TRAINING_CUDA_TESTS=1 is required even on GPU hosts. Modal
    rejects any skipped case; ordinary CPU test commands never allocate GPUs.
    """

    def setUp(self):
        if os.environ.get("REMNANT_RUN_TRAINING_CUDA_TESTS") != "1":
            self.skipTest("explicit training CUDA validation opt-in required")
        if not torch.cuda.is_available():
            self.fail("CUDA validation requested but CUDA is unavailable")
        self.assertEqual(torch.cuda.get_device_capability(), (9, 0), "SM90 only")
        self.fixture = TrainingIntegrationTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.root, self.math = self.fixture.root, self.fixture.math
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    def _record(self, name: str, result: dict) -> None:
        output = os.environ.get("REMNANT_TRAINING_VALIDATION_OUTPUT")
        if output:
            self.math.atomic_json(Path(output) / f"{name}.json", result)

    def _recipe(self, recipe: str, layer: int) -> None:
        cache = self.fixture._cache(layer)
        policy = self.fixture._warm_policy(layer, rank16=recipe == "rank64")
        for batch_size in (1, 8):
            output = self.root / f"{recipe}-{batch_size}"
            fitted = []

            def smoke_fit(*args, **kwargs):
                torch.cuda.reset_peak_memory_stats()
                result = fit(
                    *args, **{**kwargs, "epochs": 2, "train_batch_size": batch_size}
                )
                torch.cuda.synchronize()
                fitted.append((result, torch.cuda.max_memory_allocated()))
                return result

            with self.subTest(recipe=recipe, batch_size=batch_size), patch(
                "training.train.fit", side_effect=smoke_fit
            ):
                run = train.main(
                    [
                        *self.fixture._train_arguments(
                            cache, policy, recipe, str(layer), output
                        ),
                        "--device",
                        "cuda",
                    ]
                )
                result, peak = fitted[0]
                self.assertEqual(len(result["candidate_preflight"]), 2)
                self.assertEqual(result["train_batch_size"], batch_size)
                self.assertEqual(result["batch_preflight"] is None, batch_size == 1)
                selected = read_policy(Path(run["policy"]))
                for entry in selected.values():
                    read_checkpoint(entry)
                self.assertTrue(
                    json.loads((output / "progress.json").read_text())["complete"]
                )
                report = evaluate.main(
                    [
                        "--cache",
                        str(cache),
                        "--policy",
                        run["policy"],
                        "--runner",
                        str(self.fixture.runner),
                        "--output",
                        str(self.root / f"{recipe}-{batch_size}-evaluation.json"),
                        "--layers",
                        str(layer),
                        "--threads",
                        "2",
                    ]
                )
                replay = report["layers"][0]["roles"]["development"]["selected"]
                for metric in ("mean", "p95"):
                    torch.testing.assert_close(
                        torch.tensor(replay[metric], dtype=torch.float64),
                        torch.tensor(
                            result["development_selected"][metric], dtype=torch.float64
                        ),
                        atol=2e-6,
                        rtol=2e-5,
                    )
                self._record(
                    f"{recipe}-b{batch_size}",
                    {
                        "recipe": recipe,
                        "batch_size": batch_size,
                        "device": "cuda",
                        "peak_allocated_bytes": peak,
                        "history": result["history"],
                        "candidate_preflight": result["candidate_preflight"],
                        "batch_preflight": result["batch_preflight"],
                        "selected": result["development_selected"],
                        "cpu_replay": replay,
                        "policy_sha256": file_sha256(Path(run["policy"])),
                    },
                )

    def test_rank64(self):
        self._recipe("rank64", 8)

    def test_continuation(self):
        self._recipe("continuation", 8)

    def test_relaxed(self):
        self._recipe("relaxed", 8)

    def test_anchored(self):
        self._recipe("anchored", 12)

    def test_singleton_tails_and_failure_cleanup(self):
        cache = self.fixture._cache(12)
        states = evaluate.load_cached_layer(cache, 12)
        rows = [dict(states["train"][i % 4], ordinal=i) for i in range(9)]
        for mode in ("straight_through", "relaxed", "anchored"):
            counts = []
            name = (
                "forward_errors"
                if mode == "straight_through"
                else "relaxed_forward_errors"
            )
            vector = getattr(losses, name)

            def observed(math, ref_or_batch, *args, **kwargs):
                batch = args[0] if mode == "straight_through" else ref_or_batch
                counts.append(batch["raw"].shape[0])
                return vector(math, ref_or_batch, *args, **kwargs)

            with self.subTest(mode=mode), patch.object(
                losses, name, side_effect=observed
            ):
                result = fit(
                    self.math,
                    self.math.reference(),
                    rows,
                    states["development"],
                    torch.zeros(512),
                    mode="conditional_expanded",
                    epochs=2,
                    gradient_mode=mode,
                    train_batch_size=8,
                    device="cuda",
                    scratch_root=self.root,
                )
            self.assertEqual(counts[-4:], [8, 1, 8, 1])
            self.assertEqual(list(self.root.glob("remnant-anchors-*")), [])
            self._record(
                f"tail-{mode}", {"batch_counts": counts, "history": result["history"]}
            )
        with self.assertRaisesRegex(RuntimeError, "checkpoint failure"):
            fit(
                self.math,
                self.math.reference(),
                rows[:2],
                states["development"],
                torch.zeros(512),
                mode="conditional_expanded",
                epochs=2,
                gradient_mode="anchored",
                device="cuda",
                scratch_root=self.root,
                checkpoint=Mock(side_effect=RuntimeError("checkpoint failure")),
            )
        self.assertEqual(list(self.root.glob("remnant-anchors-*")), [])

    def test_staging_memory_is_batch_bounded(self):
        state = self.fixture._state("train", 0)
        peaks = []
        for count in (2, 64):
            rows = [dict(state, ordinal=i) for i in range(count)]
            with StateBatches(rows, "cuda", 2, self.root) as batches:
                batches.prepare_anchors(
                    lambda s: {
                        "state": s,
                        "rows": s["raw"] * 0.25,
                        "output": s["target"] * 0.5,
                    }
                )
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                for start in range(0, count, 2):
                    with batches.stage([start, start + 1]) as batch:
                        self.assertEqual(batches.active_count, 2)
                        self.assertTrue(all(s["raw"].is_cuda for s in batch.states))
                        self.assertTrue(all(a["rows"].is_cuda for a in batch.anchors))
                        self.assertTrue(
                            all(s["raw"].device.type == "cpu" for s in rows)
                        )
                    self.assertEqual(batches.active_count, 0)
                torch.cuda.synchronize()
                peaks.append(torch.cuda.max_memory_allocated() - baseline)
            self.assertEqual(list(self.root.glob("remnant-anchors-*")), [])
        # This checks live tensor allocations, not allocator-reserved bytes/RSS.
        self.assertLessEqual(peaks[1], peaks[0] + 2 * 1024**2)
        self._record("memory", {"state_counts": [2, 64], "peak_batch_bytes": peaks})


class PublishedCaptureTests(unittest.TestCase):
    """Opt-in checks on real publications, never synthetic capture substitutes."""

    def test_native_publication_round_trip(self):
        root = os.environ.get("REMNANT_CAPTURE_VALIDATION_ROOT")
        if not root:
            self.skipTest("real capture publication root required")
        root = Path(root)
        runner = Path(os.environ["REMNANT_CAPTURE_VALIDATION_RUNNER"])
        config = json.loads((root / "capture-config.json").read_text())
        self.assertEqual(config["storage_schema_version"], 2)
        self.assertFalse(
            config.get("conversion_only", False), "live collection required"
        )
        layers = parse_layers(",".join(map(str, config["layers"])))
        issues = sorted(config["requests"])
        reader = CaptureReader(runner, root, issues)
        records = []
        with tempfile.TemporaryDirectory(prefix="remnant-capture-check-") as scratch:
            for rid in issues:
                views = reader.load_views(Path(scratch), rid, layers)
                self.assertEqual(set(views), set(layers))
                for layer, states in views.items():
                    self.assertEqual(len(states), config["requests"][rid]["quota"])
                    self.assertTrue(all(s["q"].shape == (64, 512) for s in states))
                    records.append(
                        {"issue": rid, "layer": layer, "queries": len(states)}
                    )
            proof = None
            if config["audit_shared"]:
                proof = load_runner(runner).audit_published_capture(
                    root, Path(scratch)
                )
        output = os.environ.get("REMNANT_TRAINING_VALIDATION_OUTPUT")
        if output:
            load_runner(runner).atomic_json(
                Path(output) / "capture.json",
                {
                    "audit_shared": config["audit_shared"],
                    "records": records,
                    "audit_proof": proof,
                },
            )


if __name__ == "__main__":
    unittest.main()
