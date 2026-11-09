"""Local CPU checks for the thin Modal launchers. Author: Winston Cai.

Run: python -m unittest scripts.modal.tests
No image builds, remote calls, weights or GPUs. Build/execution boundaries are mocked.
"""

import ast
import copy
import inspect
import io
import json
import os
import runpy
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from scripts import build_lock, validation_tier
from scripts.modal import kernel_qualification as qualification
from scripts.modal import kernels
from scripts.sources import check_sources, integration_profile


def profiler_capabilities(argv, *args, **kwargs):
    """Mock only the external profiler's capability queries, not its guards."""
    if argv == ["ncu", "--list-sections"]:
        return " ".join(kernels.NCU_SECTIONS)
    if "--query-metrics-mode" in argv:
        return kernels.PREFILL_METRICS
    return ""


class SourcePreflightTests(unittest.TestCase):
    """Real Git fixtures; no Docker, Modal API or shared fork mutations."""

    def setUp(self):
        # Disposable source fixtures describe the default integration, even when
        # the launcher's real-checkout checks run under the v0.5.22 profile.
        profile = patch.dict(os.environ, {"REMNANT_INTEGRATION": "v0.5.18"})
        profile.start()
        self.addCleanup(profile.stop)
        temporary = tempfile.TemporaryDirectory(prefix="remnant-sources-")
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name) / "parent with spaces"
        self.initialize(self.repo)
        self.forks = {}
        for name in ("sglang", "flashmla"):
            root = self.repo / integration_profile("v0.5.18")[name]
            self.initialize(root)
            self.forks[name] = root
        self.cutlass = self.forks["flashmla"] / "csrc/cutlass"
        self.initialize(self.cutlass)
        self.pin(self.forks["flashmla"], self.cutlass)
        for root in self.forks.values():
            self.pin(self.repo, root)

    @staticmethod
    def git(root, *args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE, timeout=10,
        ).strip()

    @classmethod
    def initialize(cls, root):
        root.mkdir(parents=True, exist_ok=True)
        cls.git(root, "init", "-q")
        cls.git(root, "config", "user.name", "Source fixture")
        cls.git(root, "config", "user.email", "fixture@example.invalid")
        cls.git(root, "config", "commit.gpgsign", "false")
        (root / "source.txt").write_text("original\n")
        cls.git(root, "add", "source.txt")
        cls.git(root, "commit", "-qm", "fixture")

    @classmethod
    def pin(cls, parent, child):
        revision = cls.git(child, "rev-parse", "HEAD")
        cls.git(parent, "update-index", "--add", "--cacheinfo", f"160000,{revision},{child.relative_to(parent)}")
        cls.git(parent, "commit", "-qm", "pin source")

    def test_exact_recursive_pins_allow_parent_edits(self):
        (self.repo / "source.txt").write_text("parent documentation edit\n")
        records = check_sources(self.repo)
        self.assertEqual(set(records), {"parent", "sglang", "flashmla", "flashmla/csrc/cutlass"})
        for name in ("sglang", "flashmla", "flashmla/csrc/cutlass"):
            self.assertEqual(records[name]["commit"], records[name]["expected_commit"])
            self.assertFalse(records[name]["dirty"])

    def test_profiles_select_matching_dockerfiles(self):
        from modal._utils.docker_utils import extract_copy_command_patterns

        for version, filename in (("v0.5.18", "Dockerfile.v0.5.18"),
                                  ("v0.5.22", "Dockerfile.v0.5.22")):
            with self.subTest(version=version):
                profile = integration_profile(version)
                self.assertEqual(profile["dockerfile"], filename)
                recipe = (kernels.REPO_ROOT / filename).read_text()
                self.assertIn(f"ARG REMNANT_INTEGRATION={version}\n", recipe)
                self.assertIn(f"ARG REMNANT_BASE_IMAGE={profile['base_image']}\n", recipe)
                for name in ("sglang", "flashmla"):
                    self.assertIn(f"ARG REMNANT_{name.upper()}_SOURCE={profile[name]}\n", recipe)
                self.assertIn('CMD ["bash"]', recipe)
                self.assertNotIn("COPY ${", recipe)
                sources = extract_copy_command_patterns(recipe.splitlines())
                for name in ("sglang", "flashmla"):
                    self.assertIn(profile[name], sources)
                    self.assertIn(f"COPY {profile[name]} ", recipe)
                    self.assertIn(f'test "$REMNANT_{name.upper()}_SOURCE" = {profile[name]}', recipe)
                if version == "v0.5.22":
                    self.assertIn("cuda-python==13.0.3 cuda-bindings==13.0.3", recipe)

    def test_version_profiles_select_only_their_own_pins(self):
        profile = integration_profile("v0.5.22")
        forks = {}
        for name in ("sglang", "flashmla"):
            root = self.repo / profile[name]
            self.initialize(root)
            self.pin(self.repo, root)
            forks[name] = root
        nested = forks["flashmla"] / "csrc/cutlass"
        self.initialize(nested)
        self.pin(forks["flashmla"], nested)
        self.pin(self.repo, forks["flashmla"])
        (self.forks["sglang"] / "source.txt").write_text("unrelated old experiment")
        records = check_sources(self.repo, integration="v0.5.22")
        self.assertEqual(records["sglang"]["path"], str(forks["sglang"].resolve()))
        self.assertEqual(records["flashmla"]["integration"], "v0.5.22")
        self.assertIn("flashmla/csrc/cutlass", records)
        with self.assertRaisesRegex(RuntimeError, "sglang is dirty"):
            check_sources(self.repo, integration="v0.5.18")
        (nested / "source.txt").write_text("unapproved dependency")
        with self.assertRaisesRegex(RuntimeError, "flashmla/csrc/cutlass is dirty"):
            check_sources(self.repo, integration="v0.5.22")

    def test_rejects_unknown_profile_and_mixed_flashmla_override(self):
        with self.assertRaises(ValueError):
            integration_profile("latest")
        legacy = self.forks["flashmla"] / "csrc/sm90/decode/sparse_fp8/config.h"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("// legacy layout\n")
        for name in ("sglang", "flashmla"):
            root = self.repo / integration_profile("v0.5.22")[name]
            self.initialize(root)
            self.pin(self.repo, root)
        with self.assertRaisesRegex(RuntimeError, "layout does not match"):
            check_sources(self.repo, {"flashmla": self.forks["flashmla"]}, integration="v0.5.22")

    def test_rejects_mixed_sglang_override_with_dynamic_version(self):
        for name in ("sglang", "flashmla"):
            root = self.repo / integration_profile("v0.5.22")[name]
            self.initialize(root)
            self.pin(self.repo, root)
        legacy = self.forks["sglang"]
        version = legacy / "python/sglang/version.py"
        version.parent.mkdir(parents=True)
        version.write_text("try:\n    __version__ = read_version()\nexcept Exception:\n    __version__ = '0.0.0.dev0'\n")
        with self.assertRaisesRegex(RuntimeError, "argument-group layout"):
            check_sources(self.repo, {"sglang": legacy}, integration="v0.5.22")

    def test_rejects_dirty_states_without_changing_them(self):
        root = self.forks["sglang"]
        for state in ("unstaged", "staged", "untracked"):
            with self.subTest(state=state):
                path = root / ("untracked.txt" if state == "untracked" else "source.txt")
                path.write_text(state)
                if state == "staged":
                    self.git(root, "add", str(path))
                before = self.git(root, "status", "--porcelain=v1", "--untracked-files=all")
                with self.assertRaisesRegex(RuntimeError, "sglang is dirty"):
                    check_sources(self.repo)
                self.assertEqual(before, self.git(root, "status", "--porcelain=v1", "--untracked-files=all"))
                # Repair only the disposable fixture, never the shared checkout.
                path.unlink() if state == "untracked" else path.write_text("original\n")
                self.git(root, "add", "-u")

    def test_rejects_ahead_behind_and_nested_mismatches(self):
        root = self.forks["sglang"]
        original = self.git(root, "rev-parse", "HEAD")
        self.git(root, "commit", "--allow-empty", "-qm", "ahead")
        ahead = self.git(root, "rev-parse", "HEAD")
        with self.assertRaisesRegex(RuntimeError, f"expected {original}, actual {ahead}"):
            check_sources(self.repo)
        self.pin(self.repo, root)
        self.git(root, "checkout", "-q", original)
        with self.assertRaisesRegex(RuntimeError, f"expected {ahead}, actual {original}"):
            check_sources(self.repo)
        self.git(root, "checkout", "-q", ahead)
        self.git(self.cutlass, "commit", "--allow-empty", "-qm", "nested ahead")
        with self.assertRaisesRegex(RuntimeError, "flashmla/csrc/cutlass pin mismatch"):
            check_sources(self.repo)

    def test_rejects_uninitialized_submodule(self):
        root = self.forks["sglang"]
        root.rename(root.with_name("uninitialized-sglang-backup"))
        root.mkdir()
        with self.assertRaisesRegex(RuntimeError, "not an initialized Git checkout"):
            check_sources(self.repo)

    def test_explicit_override_records_dirty_content_and_nested_identity(self):
        root = self.forks["flashmla"]
        first = check_sources(self.repo, {"flashmla": root})
        (root / "source.txt").write_text("candidate\n")
        second = check_sources(self.repo, {"flashmla": root})
        record = second["flashmla"]
        self.assertTrue(record["override"])
        self.assertTrue(record["dirty"])
        self.assertEqual(record["commit"], record["expected_commit"])
        self.assertNotEqual(first["flashmla"]["sha256"], record["sha256"])
        self.assertEqual(len(second["flashmla/csrc/cutlass"]["sha256"]), 64)
        (self.forks["sglang"] / "source.txt").write_text("unapproved\n")
        with self.assertRaisesRegex(RuntimeError, "sglang is dirty"):
            check_sources(self.repo, {"flashmla": root})

    def test_local_and_modal_boundaries_fail_before_external_work(self):
        root = self.forks["sglang"]
        (root / "source.txt").write_text("dirty\n")
        tools = self.repo / "mock-tools"
        tools.mkdir()
        for name in ("docker", "curl", "ssh", "lm-eval"):
            path = tools / name
            path.write_text("#!/bin/sh\necho 'unexpected external work' >&2\nexit 97\n")
            path.chmod(0o755)
        environment = {
            **os.environ, "HOST_REPO": str(self.repo), "BENCH_EXECUTION": "docker",
            "LOG_HOST": str(self.repo / "logs"), "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        }
        for script, args in (
            ("container.sh", ["rebuild"]), ("serve.sh", ["native"]),
            ("bench-serving.sh", ["fixture"]), ("bench-lswb.sh", ["fixture"]),
            ("bench-agentic.sh", ["sangfor", "fixture"]), ("bench-lb2.sh", ["fixture"]),
        ):
            with self.subTest(script=script):
                result = subprocess.run(
                    ["bash", str(kernels.REPO_ROOT / "scripts/local" / script), *args],
                    env=environment,
                    capture_output=True, text=True, timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("sglang is dirty", result.stderr)
                self.assertNotIn("unexpected external work", result.stdout + result.stderr)
        for name, status in (("docker", 0), ("curl", 1), ("sleep", 0)):
            path = tools / name
            path.write_text(f"#!/bin/sh\nexit {status}\n")
            path.chmod(0o755)
        stopped = subprocess.run(
            ["bash", str(kernels.REPO_ROOT / "scripts/local/serve.sh"), "native", "stop"],
            env=environment, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
        self.assertIn("stopped native", stopped.stdout)
        for module in ("kernels", "serve"):
            # Execute only the existing launcher's host preamble. Reaching App
            # construction is a failure: bad sources must not reach image setup.
            code = f"""
import modal, runpy
from unittest.mock import patch
with patch('scripts.sources.check_sources', side_effect=RuntimeError('source rejected')), \\
     patch.object(modal, 'App', side_effect=AssertionError('image setup reached')):
    try:
        runpy.run_module('scripts.modal.{module}', run_name='preflight_fixture')
    except RuntimeError as error:
        assert str(error) == 'source rejected', str(error)
    else:
        raise AssertionError('source gate missing')
"""
            subprocess.run([sys.executable, "-c", code], cwd=kernels.REPO_ROOT, check=True, timeout=15)


class ContractTests(unittest.TestCase):
    def test_profile_matrix_is_complete(self):
        matrix = kernels._benchmark_cases().selected_profile_matrix()
        newer = kernels.INTEGRATION["version"] == "v0.5.22"
        batches = (8, 16, 24, 28, 32) if newer else (8, 16)
        lengths = (512,) if newer else (512, 317)
        expected = {
            (f"h{head}_b{batch}_k{length}", mode)
            for head in (64, 128)
            for batch in batches
            for length in lengths
            for mode in ("native", "triton", "kernel")
        }
        self.assertEqual(len(matrix), len(expected) + 6)
        self.assertEqual(len(set(matrix)), len(matrix))
        self.assertEqual(
            {(name, mode) for suite, name, mode in matrix if suite == "decode"},
            expected,
        )
        self.assertEqual({mode for _, _, mode in matrix}, {"native", "triton", "kernel"})

    def test_representative_profile_matrix(self):
        newer = kernels.INTEGRATION["version"] == "v0.5.22"
        cases = (
            "h64_b24_k512,h64_b28_k512,h128_b32_k512,decode-r24-all_valid,prefill-r8192-all_valid"
            if newer else
            "h64_b16_k512,h64_b16_k317,decode-r24-all_valid,prefill-r8192-all_valid"
        )
        matrix = kernels._benchmark_cases().selected_profile_matrix(cases)
        self.assertEqual(len(matrix), 3 * len(cases.split(",")))
        self.assertEqual({name for _, name, _ in matrix}, set(cases.split(",")))
        for name in cases.split(","):
            self.assertEqual(
                {mode for _, case, mode in matrix if case == name},
                {"native", "triton", "kernel"},
            )
        invalid_cases = ["unknown", "h64_b16_k512,h64_b16_k512"]
        if newer:
            invalid_cases.append("h64_b16_k317")
        for invalid in invalid_cases:
            with self.assertRaisesRegex(ValueError, "unknown or duplicate"):
                kernels._benchmark_cases().selected_profile_matrix(invalid)

    def metric_csv(self, missing=None, value="1"):
        import csv
        import io

        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
        for metric in [
            "gpu__time_duration.sum",
            "dram__bytes.sum",
            "lts__t_sector_hit_rate.pct",
            "sm__warps_active.avg",
            "launch__registers_per_thread",
            "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
            "smsp__warp_issue_stalled_long_scoreboard",
        ]:
            if metric != missing:
                writer.writerow(["0", "flash_splitkv", metric, "unit", value])
        return out.getvalue()

    def test_profile_coverage_and_metric_failures(self):
        result = kernels._benchmark_cases().profile_summary(self.metric_csv(), ["splitkv"])
        self.assertEqual(len(result["metrics"]), 7)
        self.assertFalse(result["acceptance_timing"])
        for raw, required in [
            (self.metric_csv(), ["combine"]),
            (self.metric_csv("dram__bytes.sum"), ["splitkv"]),
            (self.metric_csv(value="nan"), ["splitkv"]),
            ("empty", ["splitkv"]),
        ]:
            with self.assertRaises(RuntimeError):
                kernels._benchmark_cases().profile_summary(raw, required)

    def test_wide_profile_export(self):
        import csv
        import io

        rows = list(csv.reader(io.StringIO(self.metric_csv())))
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["ID", "Kernel Name"] + [r[2] for r in rows[1:]])
        writer.writerow(["", ""] + [r[3] for r in rows[1:]])
        writer.writerow(["0", "flash_splitkv"] + [r[4] for r in rows[1:]])
        result = kernels._benchmark_cases().profile_summary(out.getvalue(), ["splitkv"])
        self.assertEqual(len(result["metrics"]), 7)
        plural = out.getvalue().replace("warp_issue_stalled", "warps_issue_stalled")
        self.assertEqual(
            len(kernels._benchmark_cases().profile_summary(plural, ["splitkv"])["metrics"]),
            7,
        )
        with self.assertRaises(RuntimeError):
            kernels._benchmark_cases().profile_summary(out.getvalue().replace(",1", ",nan"), ["splitkv"])

    def test_timing_rejects_missing_duplicate_and_invalid_cases(self):
        import importlib.util
        import sys

        path = kernels.SGLANG_SOURCE / "python/sglang/benchmark/remnant/cases.py"
        spec = importlib.util.spec_from_file_location("_schema_fixture", path)
        contract = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = contract
        spec.loader.exec_module(contract)
        records = [
            {
                "suite": suite,
                "name": name,
                "valid": True,
                "paths": {mode: {"samples": [1.0] * 30} for mode in contract.MODES},
            }
            for suite, name in contract.profile_matrix()
            if suite == "decode"
        ]
        records += [
            {
                "suite": "store",
                "name": c.name,
                "valid": True,
                "paths": {mode: {"samples": [1.0] * 30} for mode in contract.MODES},
            }
            for c in contract.store_cases()
        ]
        payload = {
            "schema_version": 2,
            "protocol": contract.PROTOCOL,
            "status": "passed",
            "cases": records,
        }
        kernels._benchmark_cases().validate_timing(payload)
        for alter in (
            lambda p: p["cases"].pop(),
            lambda p: p["cases"].append(p["cases"][0]),
            lambda p: p["cases"][0].update(valid=False),
            lambda p: p["cases"][0]["paths"].pop("kernel"),
        ):
            bad = copy.deepcopy(payload)
            alter(bad)
            with self.assertRaises(RuntimeError):
                kernels._benchmark_cases().validate_timing(bad)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        volume = patch.object(kernels, "results_volume")
        self.results_volume = volume.start()
        self.addCleanup(volume.stop)

    def test_only_three_remote_functions_and_no_model_volumes(self):
        self.assertEqual(
            set(kernels.app.registered_functions),
            {
                "_benchmark_worker",
                "_preflight_fp4_timing",
                "_preflight_glm_timing",
                "profile_kernels",
                "validate_kernels",
            },
        )
        for function in kernels.app.registered_functions.values():
            self.assertNotIn("/models", function.spec.volumes)

    def test_direct_microbench_command(self):
        command = kernels._benchmark_command("--suite", "all")
        self.assertEqual(command[:2], [sys.executable, str(kernels.SGLANG_ROOT / "benchmark/remnant/microbench.py")])
        self.assertEqual(command[2:], ["--native-library", str(kernels.NATIVE_LIBRARY), "--suite", "all"])

    def test_pytest_rejects_skipped_failed_and_empty_coverage(self):
        for values, accepted in (
            ('tests="2"', True),
            ('tests="2" skipped="1"', False),
            ('tests="2" failures="1"', False),
            ('tests="0"', False),
        ):
            (self.root / "check.xml").write_text(f"<testsuites><testsuite {values}/></testsuites>")
            with patch.object(kernels, "_run_command") as command:
                if accepted:
                    kernels._run_pytest(kernels.BENCHMARK_TESTS, self.root, "check", 100)
                else:
                    with self.assertRaisesRegex(RuntimeError, "Incomplete validation"):
                        kernels._run_pytest(kernels.BENCHMARK_TESTS, self.root, "check", 100)
                argv = command.call_args.args[0]
                self.assertEqual(
                    argv[:7],
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "--disable-warnings",
                        "--tb=short",
                        str(kernels.SGLANG_ROOT / kernels.BENCHMARK_TESTS[0]),
                    ],
                )
                self.assertIn(str(kernels.SGLANG_ROOT / kernels.BENCHMARK_TESTS[0]), argv)
                self.assertNotIn("--worker", argv)

    def test_validation_budget_is_bounded_before_execution(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        parameters = inspect.signature(function).parameters
        self.assertEqual(parameters["max_seconds"].default, 1100)
        self.assertEqual(parameters["sanitizer_scope"].default, "all")
        with (
            patch.object(kernels, "_run_context") as run,
            patch.object(kernels, "VALIDATION_TIMEOUT_CAP_S", 2700),
            patch.object(kernels, "VALIDATION_TIMEOUT_S", 2700),
        ):
            for seconds in (0, 59, 2641, 2700):
                with self.assertRaisesRegex(ValueError, "max_seconds"):
                    function("fixture", max_seconds=seconds)
            run.assert_not_called()
        with patch.object(kernels, "VALIDATION_TIMEOUT_S", 120):
            with self.assertRaisesRegex(ValueError, "between 60 and 60"):
                function("fixture", max_seconds=61)

    def test_sanitizer_order_rejects_missing_duplicate_unknown_and_disabled(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        with patch.object(kernels, "_run_context") as run:
            for order in (
                "memcheck",
                "racecheck,memcheck",
                "memcheck,memcheck,synccheck",
                "racecheck,memcheck,unknown",
            ):
                with (
                    self.subTest(order=order),
                    self.assertRaisesRegex(ValueError, "exactly once"),
                ):
                    function("fixture", sanitizer=True, sanitizer_order=order)
            with self.assertRaisesRegex(ValueError, "requires --sanitizer"):
                function("fixture", sanitizer_order="racecheck,memcheck,synccheck")
            run.assert_not_called()

    def test_sanitizer_reordering_preserves_all_tools_and_checkpoints_receipts(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        order = ("racecheck", "memcheck", "synccheck")
        record = {}
        with (
            patch.object(kernels, "INTEGRATION", {"version": "v0.5.18"}),
            patch.object(
                kernels,
                "_run_context",
                return_value=nullcontext((self.root, record, 100)),
            ),
            patch.object(kernels.time, "monotonic", return_value=0),
            patch.object(kernels, "_run_pytest") as run,
        ):
            function("fixture", sanitizer=True, sanitizer_order=",".join(order))
        self.assertEqual(
            [c.args[2] for c in run.call_args_list], ["validation", *order]
        )
        self.assertEqual(
            set(record["sanitizer_results"]), {"racecheck", "memcheck", "synccheck"}
        )
        self.assertTrue(
            all(r["status"] == "passed" for r in record["sanitizer_results"].values())
        )
        self.assertEqual(
            json.loads((self.root / "result.json").read_text())["sanitizer_results"],
            record["sanitizer_results"],
        )
        self.assertEqual(self.results_volume.commit.call_count, 6)

    def test_sanitizer_failures_preserve_later_results_and_fail_the_run(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        order = ("synccheck", "racecheck", "memcheck")
        for failed in (None, *order):
            record = {}

            def run_pytest(tests, root, name, deadline, **kwargs):
                if name == failed:
                    raise RuntimeError(f"{name}: hazard or incomplete coverage")

            with (
                self.subTest(failed=failed),
                patch.object(kernels, "INTEGRATION", {"version": "v0.5.18"}),
                patch.object(kernels, "SANITIZER_ORDER", order),
                patch.object(kernels, "_run_context", return_value=nullcontext((self.root, record, 100))),
                patch.object(kernels.time, "monotonic", return_value=0),
                patch.object(kernels, "_run_pytest", side_effect=run_pytest) as run,
            ):
                if failed is None:
                    self.assertEqual(function("fixture", sanitizer=True), str(self.root))
                else:
                    with self.assertRaisesRegex(RuntimeError, "qualification incomplete"):
                        function("fixture", sanitizer=True)
            self.assertEqual([c.args[2] for c in run.call_args_list], ["validation", *order])
            self.assertTrue(record["sanitizer"])
            for tool in order:
                self.assertEqual(record["sanitizer_results"][tool]["status"],
                                 "failed" if tool == failed else "passed")
            for c in run.call_args_list[1:]:
                self.assertEqual(c.args[0], kernels.SANITIZER_TESTS)
                self.assertEqual(c.kwargs["prefix"][-2:], ("--error-exitcode", "1"))

    def test_sanitizer_deadline_keeps_unstarted_tools_explicitly_incomplete(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        for now in (99.5, 100):
            record = {}
            with (
                self.subTest(now=now),
                patch.object(kernels, "INTEGRATION", {"version": "v0.5.18"}),
                patch.object(kernels, "SANITIZER_ORDER", ("synccheck", "racecheck", "memcheck")),
                patch.object(kernels, "_run_context", return_value=nullcontext((self.root, record, 100))),
                patch.object(kernels.time, "monotonic", side_effect=[0, now]),
                patch.object(kernels, "_run_pytest") as run,
            ):
                with self.assertRaisesRegex(RuntimeError, "qualification incomplete"):
                    function("fixture", sanitizer=True)
            self.assertEqual([c.args[2] for c in run.call_args_list], ["validation", "synccheck"])
            self.assertEqual(record["sanitizer_results"], {
                "synccheck": {"status": "passed", "report_schema": "remnant-sanitizer-raw-v1"},
                "racecheck": {"status": "not_run"},
                "memcheck": {"status": "not_run"},
            })

    def test_sanitizer_prelaunch_timeout_keeps_tool_not_run(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        run_pytest = kernels._run_pytest
        record = {}

        def validate_or_run(tests, root, name, deadline, **kwargs):
            if name != "validation":
                return run_pytest(tests, root, name, deadline, **kwargs)

        with (
            patch.object(kernels, "INTEGRATION", {"version": "v0.5.18"}),
            patch.object(kernels, "SANITIZER_ORDER", ("synccheck", "racecheck", "memcheck")),
            patch.object(kernels, "_run_context", return_value=nullcontext((self.root, record, 100))),
            patch.object(kernels.time, "monotonic", side_effect=[99.0, 99.01]),
            patch.object(kernels, "_run_pytest", side_effect=validate_or_run) as run,
            patch.object(kernels.subprocess, "Popen") as command,
        ):
            with self.assertRaisesRegex(RuntimeError, "qualification incomplete"):
                function("fixture", sanitizer=True)
        command.assert_not_called()
        self.assertEqual([c.args[2] for c in run.call_args_list], ["validation", "synccheck"])
        self.assertEqual(record["sanitizer_results"], {
            tool: {"status": "not_run"} for tool in ("synccheck", "racecheck", "memcheck")
        })

    def test_sanitizer_tool_selection_parsing(self):
        available = ("synccheck", "racecheck", "memcheck")
        self.assertEqual(kernels._select_sanitizer_tools("", available), available)
        self.assertEqual(kernels._select_sanitizer_tools("  ", available), available)
        self.assertEqual(kernels._select_sanitizer_tools("memcheck", available), ("memcheck",))
        self.assertEqual(
            kernels._select_sanitizer_tools("memcheck, synccheck", available), ("memcheck", "synccheck")
        )
        for bad in ("initcheck", "memcheck,memcheck", "memcheck,", ",", "memcheck,racecheck,bogus"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "sanitizer_tools"):
                kernels._select_sanitizer_tools(bad, available)

    def test_sanitizer_tool_subset_runs_only_those_tools_and_is_marked_incomplete(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        order = ("synccheck", "racecheck", "memcheck")
        cases = (
            ("", order, True),
            ("memcheck,synccheck,racecheck", ("memcheck", "synccheck", "racecheck"), True),
            ("racecheck", ("racecheck",), False),
            ("memcheck,synccheck", ("memcheck", "synccheck"), False),
        )
        for spec, ran, complete in cases:
            record = {}
            with (
                self.subTest(spec=spec),
                patch.object(kernels, "INTEGRATION", {"version": "v0.5.18"}),
                patch.object(kernels, "SANITIZER_ORDER", order),
                patch.object(kernels, "_run_context", return_value=nullcontext((self.root, record, 100))),
                patch.object(kernels.time, "monotonic", return_value=0),
                patch.object(kernels, "_run_pytest") as run,
            ):
                function("fixture", sanitizer=True, sanitizer_tools=spec)
            self.assertEqual([c.args[2] for c in run.call_args_list], ["validation", *ran])
            self.assertEqual(record["sanitizer_tools"], list(ran))
            self.assertEqual(record["sanitizer_qualification"], "complete" if complete else "partial")
            self.assertEqual(
                {tool: state["status"] for tool, state in record["sanitizer_results"].items()},
                {tool: "passed" if tool in ran else "not_selected" for tool in order},
            )

    def test_sanitizer_subset_failure_still_fails_the_run_and_names_only_selected_tools(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        order = ("synccheck", "racecheck", "memcheck")
        record = {}

        def run_pytest(tests, root, name, deadline, **kwargs):
            if name == "racecheck":
                raise RuntimeError("racecheck: hazard")

        with (
            patch.object(kernels, "INTEGRATION", {"version": "v0.5.18"}),
            patch.object(kernels, "SANITIZER_ORDER", order),
            patch.object(kernels, "_run_context", return_value=nullcontext((self.root, record, 100))),
            patch.object(kernels.time, "monotonic", return_value=0),
            patch.object(kernels, "_run_pytest", side_effect=run_pytest),
        ):
            with self.assertRaisesRegex(RuntimeError, r"incomplete: \['racecheck'\]"):
                function("fixture", sanitizer=True, sanitizer_tools="racecheck")
        self.assertEqual(record["sanitizer_results"]["racecheck"]["status"], "failed")
        self.assertEqual(record["sanitizer_results"]["memcheck"], {"status": "not_selected"})
        self.assertEqual(record["sanitizer_qualification"], "incomplete")

    def test_sanitizer_subset_deadline_leaves_selected_tool_not_run_and_fails(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        record = {}
        with (
            patch.object(kernels, "INTEGRATION", {"version": "v0.5.18"}),
            patch.object(kernels, "SANITIZER_ORDER", ("synccheck", "racecheck", "memcheck")),
            patch.object(kernels, "_run_context", return_value=nullcontext((self.root, record, 100))),
            patch.object(kernels.time, "monotonic", return_value=100),
            patch.object(kernels, "_run_pytest") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, r"incomplete: \['memcheck'\]"):
                function("fixture", sanitizer=True, sanitizer_tools="memcheck")
        self.assertEqual([c.args[2] for c in run.call_args_list], ["validation"])
        self.assertEqual(record["sanitizer_results"]["memcheck"], {"status": "not_run"})
        self.assertEqual(record["sanitizer_qualification"], "incomplete")

    def test_sanitizer_tools_require_sanitizer_and_are_checked_before_execution(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        with patch.object(kernels, "_run_context") as run:
            with self.assertRaisesRegex(ValueError, "requires sanitizer"):
                function("fixture", sanitizer=False, sanitizer_tools="memcheck")
            with self.assertRaisesRegex(ValueError, "sanitizer_tools must be"):
                function("fixture", sanitizer=True, sanitizer_tools="initcheck")
            run.assert_not_called()

    def test_binding_check_validates_both_directions_without_unrelated_repairs(self):
        versions = {"cuda-python": "13.0.3", "cuda-bindings": "13.0.3", "cuda-pathfinder": "1.8.3"}
        distributions = [
            SimpleNamespace(metadata={"Name": "cuda-python"}, requires=["cuda-bindings~=13.0.3", "cuda-pathfinder~=1.1"]),
            SimpleNamespace(metadata={"Name": "torch"}, requires=["cuda-bindings>=13.0.3,<14"]),
            SimpleNamespace(metadata={"Name": "unrelated-video"}, requires=["missing-opencv>=1"]),
        ]
        with (
            patch.object(kernels.importlib.metadata, "distributions", return_value=distributions),
            patch.object(kernels.importlib.metadata, "version", side_effect=lambda name: versions[name]),
        ):
            self.assertEqual(len(kernels._check_cuda_binding_requirements()), 3)
            versions["cuda-pathfinder"] = "0.9"
            with self.assertRaisesRegex(RuntimeError, "cuda-python: requires cuda-pathfinder"):
                kernels._check_cuda_binding_requirements()
            versions["cuda-pathfinder"] = "1.8.3"
            distributions[1].requires = ["cuda-bindings>=13.4"]
            with self.assertRaisesRegex(RuntimeError, "torch: requires cuda-bindings"):
                kernels._check_cuda_binding_requirements()
            versions["cuda-bindings"] = "13.4.4"
            with self.assertRaisesRegex(RuntimeError, "requires cuda-bindings==13.0.3"):
                kernels._check_cuda_binding_requirements()

    def test_source_fingerprint_ignores_regenerable_caches(self):
        sources = [self.root / name for name in ("sglang", "flashmla")]
        for source in sources:
            source.mkdir()
            (source / "runtime.py").write_text("original")
        with (
            patch.object(kernels, "SGLANG_ROOT", sources[0]),
            patch.object(kernels, "FLASHMLA_ROOT", sources[1]),
        ):
            before = kernels._source_hashes()
            cache = sources[0] / ".pytest_cache"
            cache.mkdir()
            (cache / "nodeids").write_text("collected cases")
            self.assertEqual(kernels._source_hashes(), before)
            (sources[0] / "runtime.py").write_text("changed")
            self.assertNotEqual(kernels._source_hashes(), before)

    def test_collection_and_sanitizer_use_existing_pytest(self):
        with patch.object(kernels, "_run_command") as command:
            kernels._run_pytest(kernels.VALIDATION_TESTS, self.root, "collect", 100, collect=True)
            self.assertIn("--collect-only", command.call_args.args[0])
            (self.root / "memcheck.xml").write_text('<testsuite tests="1"/>')
            prefix = ("compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "1")
            command.return_value = "========= ERROR SUMMARY: 0 errors\n"
            kernels._run_pytest(kernels.SANITIZER_TESTS, self.root, "memcheck", 100, prefix=prefix)
            self.assertEqual(command.call_args.args[0][:5], list(prefix))

    def test_sanitizer_requires_clean_summary_and_actual_instrumentation(self):
        for tool in ("memcheck", "synccheck", "racecheck"):
            summary = ("========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
                       if tool == "racecheck" else "========= ERROR SUMMARY: 0 errors\n")
            (self.root / f"{tool}.xml").write_text('<testsuite tests="1"/>')
            for text, accepted in ((summary, True), ("....................", False),
                                   (summary + "========= Warning: overflow of tracked barriers\n", False),
                                   (".========= Warning: overflow of tracked barriers\n" + summary, False),
                                   (summary + "========= ERROR SUMMARY: 2 errors\n", False),
                                   ("========= ERROR SUMMARY: 2 errors\n" + summary, False),
                                   (summary + "========= Target application terminated before first instrumented API call\n", False)):
                with (
                    self.subTest(tool=tool, text=text),
                    patch.object(kernels, "_run_command", return_value=text),
                ):
                    kwargs = {"prefix": ("compute-sanitizer", "--tool", tool, "--error-exitcode", "1")}
                    if accepted:
                        kernels._run_pytest(kernels.SANITIZER_TESTS, self.root, tool, 100, **kwargs)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "Incomplete sanitizer evidence"):
                            kernels._run_pytest(kernels.SANITIZER_TESTS, self.root, tool, 100, **kwargs)

    def test_missing_or_malformed_pytest_report_is_an_explicit_failure(self):
        for content in (None, "<testsuites>", '<testsuite tests="invalid"/>'):
            report = self.root / "missing.xml"
            if content is not None:
                report.write_text(content)
            with (
                self.subTest(content=content),
                patch.object(kernels, "_run_command", return_value=""),
            ):
                with self.assertRaisesRegex(RuntimeError, "Incomplete validation missing"):
                    kernels._run_pytest(kernels.VALIDATION_TESTS, self.root, "missing", 100)

    def test_command_failures_show_log_path_tail_and_cause(self):
        with patch.object(kernels.subprocess, "Popen") as process:
            with self.assertRaises(TimeoutError):
                kernels._run_command(["command"], self.root / "output", 0)
            process.assert_not_called()
        with patch.object(kernels, "SGLANG_ROOT", self.root):
            with self.assertRaisesRegex(RuntimeError, "specific compiler failure") as raised:
                kernels._run_command(
                    [sys.executable, "-c", "print('specific compiler failure'); raise SystemExit(7)"],
                    self.root / "output", kernels.time.monotonic() + 30,
                )
            self.assertIn(str(self.root / "output"), str(raised.exception))
            self.assertIsInstance(raised.exception.__cause__, subprocess.CalledProcessError)
            self.assertEqual(raised.exception.__cause__.returncode, 7)
            with self.assertRaises(RuntimeError) as missing:
                kernels._run_command(
                    [str(self.root / "absent")], self.root / "missing-command",
                    kernels.time.monotonic() + 30,
                )
            self.assertIsInstance(missing.exception.__cause__, FileNotFoundError)

    def test_command_timeout_stops_grandchild_after_leader_exits(self):
        heartbeat = self.root / "heartbeat"
        child = (
            "import signal,sys,time\nfrom pathlib import Path\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\np=Path(sys.argv[1])\n"
            "p.write_text('ready\\n')\nwhile True:\n"
            "    with p.open('a') as stream: stream.write('alive\\n')\n"
            "    time.sleep(.02)\n"
        )
        leader = (
            "import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); "
            "print('compiler workers started',flush=True); time.sleep(60)"
        )
        launched = []
        real_popen = subprocess.Popen

        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            launched.append(process)
            return process

        try:
            with (
                patch.object(kernels, "SGLANG_ROOT", self.root),
                patch.object(kernels.subprocess, "Popen", side_effect=launch) as start,
                self.assertRaisesRegex(RuntimeError, "compiler workers started") as raised,
            ):
                kernels._run_command(
                    [sys.executable, "-c", leader, child, str(heartbeat)],
                    self.root / "timeout.log", kernels.time.monotonic() + 2,
                )
            self.assertTrue(start.call_args.kwargs["start_new_session"])
            self.assertIsInstance(raised.exception.__cause__, subprocess.TimeoutExpired)
            self.assertLessEqual(raised.exception.__cause__.timeout, 2)
            self.assertIsNotNone(launched[0].returncode)
            self.assertIn("ready", heartbeat.read_text())
            size = heartbeat.stat().st_size
            kernels.time.sleep(.2)
            self.assertEqual(heartbeat.stat().st_size, size)
        finally:
            for process in launched:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)

    def test_command_interruption_cleans_group_and_preserves_interrupt(self):
        process = MagicMock(pid=12345)
        interruption = KeyboardInterrupt()
        process.wait.side_effect = [interruption, 0, 0]
        with (
            patch.object(kernels.subprocess, "Popen", return_value=process),
            patch.object(kernels.os, "killpg") as kill,
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            kernels._run_command(["compiler"], self.root / "interrupted.log",
                                 kernels.time.monotonic() + 30)
        self.assertIs(raised.exception, interruption)
        self.assertEqual(kill.call_args_list, [call(12345, signal.SIGTERM), call(12345, signal.SIGKILL)])
        self.assertEqual(process.wait.call_args_list[-2:], [call(timeout=5), call(timeout=5)])

    def test_command_cleanup_escalates_and_reports_failure(self):
        process = MagicMock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired("compiler", 5), 0]
        with patch.object(kernels.os, "killpg") as kill:
            kernels._stop_command_group(process)
        self.assertEqual(kill.call_args_list, [call(12345, signal.SIGTERM), call(12345, signal.SIGKILL)])
        with (
            patch.object(kernels.os, "killpg", side_effect=PermissionError("denied")),
            self.assertRaises(PermissionError),
        ):
            kernels._stop_command_group(process)

    def test_command_cleanup_failure_retains_original_exit_status(self):
        process = MagicMock(pid=12345)
        process.wait.return_value = 7
        cleanup_error = PermissionError("denied")
        with (
            patch.object(kernels.subprocess, "Popen", return_value=process),
            patch.object(kernels, "_stop_command_group", side_effect=cleanup_error),
            self.assertRaises(RuntimeError) as raised,
        ):
            kernels._run_command(["compiler"], self.root / "cleanup-failure.log",
                                 kernels.time.monotonic() + 30)
        failure = raised.exception.__cause__
        self.assertIsInstance(failure, subprocess.CalledProcessError)
        self.assertEqual(failure.returncode, 7)
        self.assertIs(failure.__cause__, cleanup_error)
        self.assertIn("Process group cleanup failed", failure.__notes__[0])

    def test_records_survive_failure_and_ids_cannot_overwrite(self):
        manifest = self.root / "image.json"
        manifest.write_text('{"sources":{"sglang":"hash"}}')
        with (
            patch.object(kernels, "RESULTS_ROOT", self.root),
            patch.object(kernels, "MANIFEST", manifest),
            patch.object(kernels.results_volume, "commit") as results,
            patch.object(kernels.triton_volume, "commit") as cache,
        ):
            with self.assertRaisesRegex(RuntimeError, "example"), kernels._run_context("failure", "timing", 30):
                raise RuntimeError("example")
            saved = self.root / "microbench/failure/timing/result.json"
            self.assertEqual(json.loads(saved.read_text())["status"], "failed")
            results.assert_called_once()
            cache.assert_called_once()
            with self.assertRaises(FileExistsError), kernels._run_context("failure", "timing", 30):
                pass
            with self.assertRaises(ValueError), kernels._run_context("../escape", "timing", 30):
                pass

    def test_uploads_include_tests_and_exclude_binaries(self):
        for name in (
            *kernels.VALIDATION_TESTS,
            *kernels.BENCHMARK_TESTS,
            "test/registered/attention/unittests/conftest.py",
            "python/sglang/kernels/aot/CMakeLists.txt",
        ):
            self.assertFalse(kernels._sglang_ignored(Path(name)), name)
        for name in (
            "examples/demo.py",
            "python/sglang/__pycache__/test.pyc",
            "python/sglang/kernels/aot/build/test.so",
        ):
            self.assertTrue(kernels._sglang_ignored(Path(name)), name)
        self.assertFalse(kernels._flashmla_ignored(Path("csrc/detail.cuh")))
        self.assertTrue(kernels._flashmla_ignored(Path("tests/test_decode.py")))

    def test_image_build_compiles_on_cpu_then_preflights(self):
        newer = kernels.INTEGRATION["version"] == "v0.5.22"
        sglang, flashmla = self.root / "sglang", self.root / "flashmla"
        aot = sglang / "python/sglang/kernels/aot"
        package = self.root / "site/sgl_kernel"
        cache = self.root / "jit"
        library = cache / "module/key/valid/native.so"
        for directory in (aot / "python/sgl_kernel", flashmla, package, library.parent):
            directory.mkdir(parents=True)
        (aot / "python/sgl_kernel/flash_mla.py").write_text("# fork binding")
        native, manifest = self.root / "native.so", self.root / "manifest.json"
        cuda_home = self.root / "cuda"
        stub = cuda_home / "lib64/stubs/libcuda.so"
        stub.parent.mkdir(parents=True)
        stub.write_bytes(b"CPU-only driver stub")
        calls = []
        sanitizer = self.root / "sanitizer/compute-sanitizer"
        if newer:
            (package / "flash_mla.py").write_text("# fork binding")
            for target in ("remnant_ops", "flashmla_ops"):
                (package / (target + ".so")).write_bytes(target.encode())

        def command(argv, output, deadline, *, environment=None):
            calls.append(argv)
            if argv == ["ncu", "--list-sections"]:
                return " ".join(kernels.NCU_SECTIONS)
            if "--query-metrics-mode" in argv:
                return kernels.PREFILL_METRICS
            if argv[0] == "dpkg-deb":
                sanitizer.parent.mkdir(parents=True)
                sanitizer.write_bytes(b"pinned tool")
            if argv[0] == "cuobjdump":
                return "ELF file sm_90a"
            if argv[:2] == ["cmake", "--build"]:
                for target in ("remnant_ops", "flashmla_ops"):
                    binary = self.root / "build/cmake" / (target + ".so")
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_bytes(target.encode())
            if "--check-imports" in argv:
                self.assertTrue(native.is_file())
                self.assertEqual(
                    (package / "flash_mla.py").read_text(), "# fork binding"
                )
                self.assertTrue(
                    Path(
                        environment["LD_LIBRARY_PATH"].split(":")[0], "libcuda.so.1"
                    ).samefile(stub)
                )
            return "test toolchain"

        def native_factory(*args, **kwargs):
            self.assertEqual(args, ("fp32", 512, 64, 64, False))
            self.assertEqual(kwargs, {"layout": "v4"} if newer else {})
            library.write_bytes(b"native")

        modules = {
            "torch": SimpleNamespace(
                float32="fp32",
                __version__="test",
                version=SimpleNamespace(cuda="13.0"),
                utils=SimpleNamespace(cmake_prefix_path="/torch"),
            ),
            "sglang.kernels.jit.utils": SimpleNamespace(
                override_jit_cuda_arch=lambda *a: nullcontext()
            ),
            "sglang.kernels.jit.utils.compile.cache": SimpleNamespace(
                find_prebuilt=lambda **kw: library
            ),
            "sglang.kernels.ops.attention.dsv4.compress": SimpleNamespace(
                _jit_compress_norm_rope_module=native_factory
            ),
            "sglang.kernels.ops.attention.dsv4.kv_layout": SimpleNamespace(
                KVLayout=SimpleNamespace(V4="v4")
            ),
        }
        # Synthetic downloads exercise orchestration without network access.
        # Hash checks themselves have a separate changed-content rejection case.
        sha256 = kernels.hashlib.sha256
        download_hashes = {
            b"sanitizer fixture": "ab5467d9473adfb528481d1b6166b9bc718ab668563a3466d0f2c5d8dd27aa4f",
            b"config fixture": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
        }

        def digest(payload=b""):
            if payload in download_hashes:
                return SimpleNamespace(hexdigest=lambda: download_hashes[payload])
            return sha256(payload)

        def download(url, *, timeout):
            self.assertEqual(timeout, 30)
            return io.BytesIO(b"config fixture" if url.endswith("config.json") else b"sanitizer fixture")

        def build_path(value):
            if str(value) == "/opt/remnant-launcher.py":
                return kernels.REPO_ROOT / "scripts/modal/kernels.py"
            if str(value) == "/opt/remnant-0731-config.json":
                return self.root / "config.json"
            return Path(value)

        with (
            patch.dict(sys.modules, modules),
            patch.object(kernels, "SGLANG_ROOT", sglang),
            patch.object(kernels, "FLASHMLA_ROOT", flashmla),
            patch.object(kernels, "MANIFEST", manifest),
            patch.object(kernels, "NATIVE_LIBRARY", native),
            patch.object(kernels, "BUILD_ROOT", self.root / "build"),
            patch.object(kernels, "SANITIZER_ROOT", sanitizer.parent),
            patch.object(kernels, "SANITIZER", str(sanitizer)),
            patch.object(kernels, "Path", side_effect=build_path),
            patch.object(kernels.urllib.request, "urlopen", side_effect=download),
            patch.object(kernels.hashlib, "sha256", side_effect=digest),
            patch.object(kernels.importlib.metadata, "version", return_value="13.0.3"),
            patch.object(kernels.importlib.metadata, "distributions", return_value=[]),
            patch.object(
                kernels.sysconfig,
                "get_paths",
                return_value={"purelib": str(package.parent)},
            ),
            patch.dict(
                kernels.os.environ,
                {"SGLANG_JIT_CACHE_DIR": str(cache), "CUDA_HOME": str(cuda_home)},
            ),
            patch.object(kernels, "_run_command", side_effect=command),
            patch.object(kernels, "_run_pytest") as tests,
        ):
            kernels._build_image()
        if newer:
            self.assertFalse(any(c[0] == "cmake" for c in calls))
            self.assertFalse(any(c[0] == "ncu" for c in calls))
            self.assertEqual(sum(c[0] == "cuobjdump" for c in calls), 3)
        else:
            configure = next(c for c in calls if c[:2] == ["cmake", "-S"])
            self.assertIn("-DSGL_KERNEL_ENABLE_FLASHMLA_SM100=OFF", configure)
            self.assertIn("-DENABLE_BELOW_SM90=OFF", configure)
            self.assertEqual(calls[0][:2], [sys.executable, "-c"])
        self.assertEqual(tests.call_count, 4 if newer else 3)
        self.assertTrue(
            all(
                "cpu-driver-stubs" in c.kwargs["environment"]["LD_LIBRARY_PATH"]
                for c in tests.call_args_list
            )
        )
        receipt = json.loads(manifest.read_text())
        self.assertEqual(receipt["arch"], "sm_90a")
        self.assertEqual(len(receipt["binaries"]), 3)
        self.assertEqual(receipt["integration"], kernels.INTEGRATION["version"])

    def test_cpu_and_sanitizer_selection_use_names_not_list_positions(self):
        extra_cpu = ()
        if kernels.INTEGRATION["version"] == "v0.5.22":
            extra_cpu = (
                kernels.TEST_MODULES["pool"] + "::TestRemnantPool",
                "benchmark/remnant/test_microbench.py::test_packed_hicache_routes_all_planes_without_cuda",
                "benchmark/remnant/test_microbench.py::test_prefill_fixture_matches_combine_keyword_contract",
                "test/registered/unit/utils/test_nvtx_utils.py",
                "test/registered/unit/observability/test_req_time_stats.py",
            )
        self.assertEqual(
            kernels.CPU_TESTS,
            tuple(
                kernels.TEST_MODULES[n]
                for n in ("cache_format", "pack", "backend", "compressor_store")
            ) + extra_cpu,
        )
        expected = {
            kernels.TEST_MODULES["unpack_kernel"]
            + "::test_triton_native_pages_match_independent_reference",
            kernels.TEST_MODULES["cuda_graph"]
            + "::TestRemnantCudaGraph::test_store_and_native_reconstruction_replays_changed_inputs",
            kernels.TEST_MODULES["pack_kernel"]
            + "::test_remnant_triton_pack_matches_reference",
            kernels.TEST_MODULES["cuda_graph"]
            + "::TestRemnantCudaGraph::test_consecutive_mixed_layers_replay_with_stable_buffers",
            kernels.TEST_MODULES["cuda_graph"]
            + "::TestRemnantCudaGraph::test_fused_read_replays_changed_planes_indices_lengths_and_frequencies",
        }
        if kernels.INTEGRATION["version"] == "v0.5.22":
            extra_sanitizer = {
                "flashmla_direct": (
                    "TestRemnantFlashMLADirect::test_direct_matches_fused_adapter[kernel-512-8-64-topmag]",
                    "TestRemnantFlashMLADirect::test_direct_matches_fused_adapter[kernel-512-8-128-bitmap-edges]",
                    "TestRemnantFlashMLADirect::test_direct_decode_cuda_graph_replay[8-64]",
                ),
                "fused_store": ("test_graph_replay_with_changed_inputs_and_plans",),
                "hicache": ("TestRemnantHiCache::test_backup_restore_preserves_all_planes_across_layers_and_pages",),
            }
            expected.update(
                kernels.TEST_MODULES[module] + "::" + node
                for module, nodes in extra_sanitizer.items() for node in nodes
            )
        self.assertEqual(set(kernels.SANITIZER_TESTS), expected)
        self.assertEqual(len(kernels.SANITIZER_TESTS), len(expected))
        # Check actual node identities without importing CUDA-dependent modules.
        for selector in kernels.SANITIZER_TESTS:
            path, *names = selector.split("::")
            node = ast.parse((kernels.SGLANG_SOURCE / path).read_text())
            for name in names:
                name = name.split("[", 1)[0]
                matches = [
                    child for child in node.body if getattr(child, "name", None) == name
                ]
                self.assertEqual(len(matches), 1, selector)
                node = matches[0]

    def test_cuda_binding_constraints_reject_missing_or_conflicting_consumers(self):
        distribution = SimpleNamespace(
            metadata={"Name": "torch"}, requires=["cuda-bindings>=13.0.3"]
        )
        with (
            patch.object(kernels.importlib.metadata, "distributions", return_value=[distribution]),
            patch.object(kernels.importlib.metadata, "version", return_value="13.0.3"),
        ):
            self.assertEqual(kernels._check_cuda_binding_requirements()[0]["consumer"], "torch")
            distribution.requires = ["cuda-bindings>=13.4"]
            with self.assertRaisesRegex(RuntimeError, "requires cuda-bindings"):
                kernels._check_cuda_binding_requirements()
            distribution.metadata["Name"] = "cuda-python"
            distribution.requires = ["cuda-missing>=1"]
            def version(name):
                if name == "cuda-missing":
                    raise kernels.importlib.metadata.PackageNotFoundError(name)
                return "13.0.3"
            with patch.object(kernels.importlib.metadata, "version", side_effect=version):
                with self.assertRaisesRegex(RuntimeError, "missing cuda-missing"):
                    kernels._check_cuda_binding_requirements()
        with patch.object(kernels.importlib.metadata, "version", return_value="13.4"):
            with self.assertRaisesRegex(RuntimeError, "Serving image requires"):
                kernels._check_cuda_binding_requirements()

    def test_build_rejects_changed_sanitizer_download_before_compilation(self):
        with (
            patch.object(kernels, "INTEGRATION", integration_profile("v0.5.22")),
            patch.object(kernels, "BUILD_ROOT", self.root / "bad-download"),
            patch.object(kernels, "_check_cuda_binding_requirements", return_value=[]),
            patch.dict(sys.modules, {"sglang.kernels.jit.utils": SimpleNamespace(override_jit_cuda_arch=None)}),
            patch.object(kernels.urllib.request, "urlopen", return_value=io.BytesIO(b"changed package")),
            patch.object(kernels, "_run_command") as command,
        ):
            with self.assertRaisesRegex(RuntimeError, "Sanitizer package content changed"):
                kernels._build_image()
            command.assert_not_called()

    def test_profiler_capability_guards_are_not_bypassed(self):
        for response, message in (("", "required sections"), (" ".join(kernels.NCU_SECTIONS), "traffic metrics")):
            with self.subTest(response=response), patch.object(kernels, "_run_command", return_value=response):
                with self.assertRaisesRegex(RuntimeError, message):
                    kernels._check_profiler_tools(self.root)

    def test_cached_serving_image_rejects_mismatched_sources(self):
        from scripts.modal import serve

        with (
            patch("scripts.sources.compiled_source_digest", return_value="matching"),
            patch.dict(os.environ, {"REMNANT_BUILT_INTEGRATION": serve.INTEGRATION["version"]}),
        ):
            serve._verify_compiled_sources("matching")
            with self.assertRaisesRegex(RuntimeError, "Compiled source changed"):
                serve._verify_compiled_sources("different")
            wrong = "v0.5.18" if serve.INTEGRATION["version"] == "v0.5.22" else "v0.5.22"
            with patch.dict(os.environ, {"REMNANT_BUILT_INTEGRATION": wrong}):
                with self.assertRaisesRegex(RuntimeError, "different integration"):
                    serve._verify_compiled_sources("matching")
        source = self.root / "source"
        source.mkdir()
        kernel = source / "kernel.cu"
        kernel.write_text("original")
        before = serve._compiled_source_digest([source])
        kernel.write_text("changed")
        self.assertNotEqual(before, serve._compiled_source_digest([source]))

    def test_image_source_hash_ignores_git_pointers_but_not_code(self):
        roots = [self.root / name for name in ("sglang", "flashmla")]
        for root in roots:
            (root / "nested").mkdir(parents=True)
            (root / "kernel.cu").write_text("compiled source\n")
            (root / "nested/.git").write_text("gitdir: first-local-checkout\n")
        with (
            patch.object(kernels, "SGLANG_ROOT", roots[0]),
            patch.object(kernels, "FLASHMLA_ROOT", roots[1]),
        ):
            before = kernels._source_hashes()
            for root in roots:
                (root / "nested/.git").write_text("gitdir: another-local-checkout\n")
            self.assertEqual(kernels._source_hashes(), before)
            (roots[1] / "kernel.cu").write_text("changed compiled source\n")
            self.assertNotEqual(kernels._source_hashes()["flashmla"], before["flashmla"])

    def test_serving_missing_extensions_never_compiles_on_gpu(self):
        from scripts.modal import serve

        with (
            patch.object(serve, "_kernel_extensions_ready", return_value=False),
            patch.object(serve.subprocess, "run") as process,
        ):
            with self.assertRaisesRegex(RuntimeError, "GPU compilation is disabled"):
                serve.benchmark_serving.local()
            process.assert_not_called()

    def test_serving_wrapper_owns_boot_and_calls_client_with_remaining_budget(self):
        from scripts.modal import serve

        with (
            patch.object(serve, "_kernel_extensions_ready", return_value=True),
            patch.object(serve, "RESULTS_ROOT", self.root),
            patch.object(serve, "REMOTE_REPO", kernels.REPO_ROOT),
            patch.object(serve, "_running_serving_server", return_value=nullcontext()) as boot,
            patch.object(serve.subprocess, "run") as client,
            patch.object(serve.results_volume, "commit") as commit,
        ):
            result = Path(serve.benchmark_serving.local(timeout_minutes=1))
            self.assertTrue(result.is_dir())
            env = client.call_args.kwargs["env"]
            self.assertEqual(env["BENCH_EXECUTION"], "direct")
            self.assertEqual(env["SERVING_MODE"], "native")
            self.assertEqual(env["RESULTS_HOST"], str(result))
            self.assertEqual(client.call_args.args[0][-5:],
                             ["native", env["PORT"], "32768", "2048", "8"])
            self.assertGreater(client.call_args.kwargs["timeout"], 0)
            self.assertLessEqual(client.call_args.kwargs["timeout"], 60)
            self.assertTrue(client.call_args.kwargs["check"])
            self.assertEqual(boot.call_args.args[1], result / "server.log")
            self.assertTrue((result / "source-revisions.json").is_file())
            commit.assert_called_once()

    def test_serving_server_is_stopped_on_client_error_and_startup_failure(self):
        from scripts.modal import serve

        env = {"SERVING_MODE": "native", "PORT": "30211"}
        for state, deadline, error in ((None, float("inf"), RuntimeError),
                                       (1, float("inf"), RuntimeError),
                                       (None, 0, TimeoutError)):
            with (
                self.subTest(state=state, deadline=deadline),
                patch.object(serve.socket, "socket"),
                patch.object(serve.subprocess, "Popen") as launch,
                patch.object(serve.urllib.request, "urlopen") as health,
                patch.object(serve, "_stop_serving_server") as stop,
            ):
                process = launch.return_value
                process.poll.return_value = state
                health.return_value.__enter__.return_value.status = 200
                with self.assertRaises(error):
                    with serve._running_serving_server(env, self.root / "server.log", deadline):
                        raise RuntimeError("client failed")
                self.assertTrue(launch.call_args.kwargs["start_new_session"])
                stop.assert_called_once_with(process)

    def test_serving_cleanup_ignores_zombies_and_other_process_groups(self):
        from scripts.modal import serve

        process = MagicMock(pid=12345)
        with (
            patch.object(serve.os, "killpg") as kill,
            patch.object(serve.subprocess, "check_output", return_value="12345 Z\n99999 S\n"),
        ):
            serve._stop_serving_server(process)
            kill.assert_called_once_with(12345, serve.signal.SIGTERM)
            process.wait.assert_called_once_with(timeout=1)

    def test_serving_cleanup_escalates_and_does_not_hide_failure(self):
        from scripts.modal import serve

        process = MagicMock(pid=12345)
        with (
            patch.object(serve.os, "killpg") as kill,
            patch.object(serve.time, "monotonic", side_effect=[0, 11, 20, 21]),
            patch.object(serve.subprocess, "check_output", return_value="12345 Z\n"),
        ):
            serve._stop_serving_server(process)
            self.assertEqual(kill.call_args_list,
                             [call(12345, serve.signal.SIGTERM), call(12345, serve.signal.SIGKILL)])
        with (
            patch.object(serve.os, "killpg"),
            patch.object(serve.time, "monotonic", side_effect=[0, 11, 20, 26]),
        ):
            with self.assertRaisesRegex(RuntimeError, "survived SIGKILL"):
                serve._stop_serving_server(process)
        with patch.object(serve.os, "killpg", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                serve._stop_serving_server(process)

    def test_model_cache_requires_nonempty_pinned_receipts(self):
        from scripts.modal import serve

        model = self.root / "model"
        receipts = model / ".cache/huggingface/download"
        receipts.mkdir(parents=True)
        names = [f"model-{index:05}.safetensors" for index in range(48)]
        (model / "config.json").write_text(json.dumps({"num_attention_heads": 64, "compress_ratios": [4] * 61}))
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"weight-{index}": name for index, name in enumerate(names)}})
        )
        for name in names + ["config.json", "model.safetensors.index.json"]:
            if name in names:
                (model / name).write_bytes(b"shard")
            (receipts / (name + ".metadata")).write_text(serve.MODEL_REVISION + "\netag\n0\n")
        receipt = receipts / (names[0] + ".metadata")
        with patch.object(serve, "MODEL_DIR", model):
            with patch("builtins.print"):
                result = serve.inspect_model_cache.local()
            self.assertEqual(result["shards"], 48)
            self.assertEqual(result["bytes"], 48 * len(b"shard"))
            for value in ("", "another-revision\netag\n0\n"):
                receipt.write_text(value)
                with self.subTest(receipt=value), self.assertRaisesRegex(
                    RuntimeError, "missing or unpinned checkpoint file: " + names[0]
                ):
                    serve.inspect_model_cache.local()

    def test_glm_serving_chat_requires_generated_content(self):
        from scripts.modal import serve

        body = {
            "model": "glm53-flash",
            "choices": [
                {"message": {"content": "1, 2, 3, 4, 5"}, "finish_reason": "stop"}
            ],
            "usage": {"completion_tokens": 8},
        }
        for content, finish_reason, succeeds in (
            ("1, 2, 3, 4, 5", "stop", True),
            (" ", "stop", False),
            ("One, two", "length", False),
        ):
            body["choices"][0]["message"]["content"] = content
            body["choices"][0]["finish_reason"] = finish_reason
            response = io.BytesIO(json.dumps(body).encode())
            response.status = 200
            with patch.object(serve.urllib.request, "urlopen", return_value=response) as send:
                deadline = serve.time.monotonic() + 10
                if succeeds:
                    result = serve._glm_chat_request("30211", deadline)
                    self.assertEqual(result["response"], body)
                    self.assertEqual(send.call_args.args[0].get_method(), "POST")
                else:
                    with self.assertRaisesRegex(RuntimeError, "Invalid GLM chat completion"):
                        serve._glm_chat_request("30211", deadline)

    def test_public_commands_import_without_launching_jobs(self):
        code = """
from unittest.mock import patch
from pathlib import Path
import modal
with patch.object(modal.Image, 'pip_install', autospec=True, side_effect=modal.Image.pip_install) as dependencies, \\
     patch.object(modal.Function, 'remote', side_effect=AssertionError('launch')), \\
     patch.object(modal.Function, 'spawn', side_effect=AssertionError('launch')), \\
     patch.object(modal.Image, 'from_dockerfile', wraps=modal.Image.from_dockerfile) as recipe:
    from scripts.modal import serve, kernels
    expected = 'Dockerfile.' + serve.INTEGRATION['version']
    assert Path(recipe.call_args.args[0]).name == expected
    assert recipe.call_args.kwargs['build_args']['REMNANT_INTEGRATION'] == serve.INTEGRATION['version']
    assert 'benchmark_serving' in serve.app.registered_functions
    assert any('modal==' + modal.__version__ in call.args for call in dependencies.call_args_list)
    assert {'_benchmark_worker', '_preflight_fp4_timing', 'profile_kernels', 'validate_kernels'} <= set(kernels.app.registered_functions)
    expected_order = ('synccheck', 'racecheck', 'memcheck') if serve.INTEGRATION['version'] == 'v0.5.22' else ('memcheck', 'racecheck', 'synccheck')
    assert kernels.SANITIZER_ORDER == expected_order
"""
        for version in ("v0.5.18", "v0.5.22"):
            environment = dict(os.environ, REMNANT_INTEGRATION=version)
            environment.pop("REMNANT_COMPILED_IMAGE", None)
            with self.subTest(version=version):
                subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=kernels.REPO_ROOT, env=environment, check=True,
                    capture_output=True, text=True, timeout=30,
                )

    def test_launchers_import_independently(self):
        for module, other in (("serve", "kernels"), ("kernels", "serve")):
            code = f"import sys; from scripts.modal import {module}; assert 'scripts.modal.{other}' not in sys.modules"
            subprocess.run(
                [sys.executable, "-c", code],
                cwd=kernels.REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )


class LocalScriptTests(unittest.TestCase):
    def test_request_timing_probe_uses_valid_lifecycle_for_both_profiles(self):
        """Author: Winston Cai. Exercise the real probe and fork exporters on CPU."""
        launcher = (kernels.REPO_ROOT / "scripts/local/serve.sh").read_text()
        probe = launcher.split(
            "from sglang.srt.observability.req_time_stats import APIServerReqTimeStats\n", 1
        )[1].split("\n' || exit 1", 1)[0]
        for version in ("v0.5.18", "v0.5.22"):
            source = (kernels.REPO_ROOT / integration_profile(version)["sglang"] /
                      "python/sglang/srt/observability/req_time_stats.py")
            # Execute the actual exporter class without importing optional GPU
            # dependencies. Its base owns tracing, unused by this metadata probe.
            stats_class = next(node for node in ast.parse(source.read_text()).body
                               if isinstance(node, ast.ClassDef) and
                               node.name == "APIServerReqTimeStats")
            stats_class.bases = []
            module = ast.Module(body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                stats_class,
            ], type_ignores=[])
            namespace = dict(__name__=__name__, dataclass=dataclass, field=field,
                             SGLANG_FRONTEND_DIAGNOSTICS=False, inspect=inspect,
                             convert_time_to_realtime=lambda stamp: 1000.0 + stamp)
            exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
            with self.subTest(version=version), patch("builtins.print"):
                exec(probe, namespace)
                stats = namespace["stats"]
                self.assertGreater(stats.created_time, 0)
                self.assertGreater(stats.first_token_time, stats.created_time)
                self.assertEqual(stats.convert_to_output_meta_info()["first_output_ts"], 1002.0)
                # Keep rejecting an exporter that really lacks the timestamp.
                namespace["APIServerReqTimeStats"] = lambda: SimpleNamespace(
                    convert_to_output_meta_info=lambda: {})
                with self.assertRaisesRegex(AssertionError, "lacks first_output_ts"):
                    exec(probe, namespace)

    def test_container_preflight_uses_selected_argument_layout(self):
        """Run the real prep action and shell checks; fake Docker only."""
        with tempfile.TemporaryDirectory(prefix="remnant-container-prep-") as temporary:
            root = Path(temporary)
            tools = root / "bin"
            tools.mkdir()
            docker = tools / "docker"
            docker.write_text(
                f"#!{sys.executable}\n"
                "import os, subprocess, sys\n"
                "args = sys.argv[1:]\n"
                "if args[:2] == ['container', 'inspect']:\n"
                "    raise SystemExit(0)\n"
                "if args[0] == 'inspect':\n"
                "    print(os.environ['IMAGE_SOURCE_LABELS'])\n"
                "    raise SystemExit(0)\n"
                "assert args[:2] == ['exec', '-i'], args\n"
                "assert args[3:6] == ['bash', '-s', '--'], args\n"
                "raise SystemExit(subprocess.call(args[3:]))\n"
            )
            docker.chmod(0o755)
            for version in ("v0.5.18", "v0.5.22"):
                profile = integration_profile(version)
                source = kernels.REPO_ROOT / profile["sglang"] / "python/sglang/srt"
                installation = root / version
                srt = installation / "python/sglang/srt"
                srt.mkdir(parents=True)
                (srt / "server_args.py").write_bytes((source / "server_args.py").read_bytes())
                fields = srt / "server_args.py"
                if version == "v0.5.22":
                    fields = srt / "arg_groups/fields/model.py"
                    fields.parent.mkdir(parents=True)
                    fields.write_bytes((source / "arg_groups/fields/model.py").read_bytes())
                pins = [subprocess.check_output(
                    ["git", "-C", str(kernels.REPO_ROOT / profile[name]), "rev-parse", "HEAD"],
                    text=True, timeout=10,
                ).strip() for name in ("sglang", "flashmla")]
                environment = dict(
                    os.environ, REMNANT_INTEGRATION=version,
                    HOST_REPO=str(kernels.REPO_ROOT), CONTAINER="fixture",
                    SGLANG_ROOT_CT=str(installation),
                    IMAGE_SOURCE_LABELS=" ".join([version, *pins]),
                    PATH=f"{tools}:{Path(sys.executable).parent}:{os.environ['PATH']}",
                )
                def prep():
                    return subprocess.run(
                        ["bash", str(kernels.REPO_ROOT / "scripts/local/container.sh"), "prep"],
                        env=environment, text=True, capture_output=True, timeout=30,
                    )
                with self.subTest(version=version):
                    result = prep()
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("production image ready", result.stdout)
                    fields.write_text(fields.read_text().replace(
                        "kv_cache_format" if version == "v0.5.22" else "dsv4_c4_cache_implementation", "missing_implementation_flag"
                    ))
                    self.assertNotEqual(prep().returncode, 0)
                    if version == "v0.5.22":
                        # A stale legacy declaration cannot replace grouped fields.
                        fields.unlink()
                        with (srt / "server_args.py").open("a") as stream:
                            stream.write("\ndsv4_c4_cache_implementation = 'kernel'\n")
                        self.assertNotEqual(prep().returncode, 0)

    def test_capture_rejects_wrong_image_profile_before_gpu_checks(self):
        with tempfile.TemporaryDirectory(prefix="remnant-capture-image-") as temporary:
            root = Path(temporary)
            for name in ("bin", "model", "scratch", "published"):
                (root / name).mkdir()
            for name in ("capture-config.json", "replay.jsonl"):
                (root / "scratch" / name).write_text("{}\n")
            commands = {
                "docker": '#!/bin/bash\nif [[ "$*" == *"{{.Id}}"* ]]; then printf "sha256:%064d\\n" 0; else echo "$FAKE_IMAGE_PROFILE wrong-sglang wrong-flashmla"; fi\n',
                "timeout": '#!/bin/bash\nwhile [[ "$1" == --* ]]; do shift; done\nshift\nexec "$@"\n',
                "nvidia-smi": '#!/bin/bash\necho unexpected-GPU-inspection >&2\nexit 1\n',
            }
            for name, script in commands.items():
                tool = root / "bin" / name
                tool.write_text(script)
                tool.chmod(0o755)
            for version in ("v0.5.18", "v0.5.22"):
                environment = dict(os.environ, REMNANT_INTEGRATION=version,
                                   FAKE_IMAGE_PROFILE="v0.5.22" if version == "v0.5.18" else "v0.5.18",
                                   PATH=f"{root / 'bin'}:{Path(sys.executable).parent}:{os.environ['PATH']}")
                with self.subTest(version=version):
                    result = subprocess.run(
                        ["bash", str(kernels.REPO_ROOT / "scripts/local/capture-latent.sh"), "run", "fake-image",
                         str(root / "model"), str(root / "scratch"), str(root / "published"), "120"],
                        env=environment, text=True, capture_output=True, timeout=30,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Capture image source/profile mismatch", result.stderr)
                    self.assertNotIn("unexpected-GPU-inspection", result.stderr)

    def test_local_profiles_match_source_and_dockerfile_selection(self):
        for version in ("v0.5.18", "v0.5.22"):
            profile = integration_profile(version)
            environment = dict(os.environ, REMNANT_INTEGRATION=version)
            environment.pop("CONTAINER", None)
            environment.pop("IMAGE", None)
            with self.subTest(version=version):
                result = subprocess.run(
                    ["bash", "-c",
                     '. "$1"; printf "%s\\n" "$REMNANT_DOCKERFILE" "$REMNANT_BASE_IMAGE" "$CONTAINER" "$IMAGE"',
                     "profile-check", str(kernels.REPO_ROOT / "scripts/local/env.sh")],
                    env=environment, capture_output=True, text=True, check=True, timeout=10,
                )
                self.assertEqual(result.stdout.splitlines(),
                                 [profile["dockerfile"], profile["base_image"], profile["container"], f"remnant:{version}"])

    def test_async_preparation_knob_validates_before_server_replacement(self):
        """Author: Winston Cai. Exercise the real shell boundary without Docker."""
        with tempfile.TemporaryDirectory(prefix="remnant-input-preparation-") as temporary:
            root = Path(temporary)
            tools = root / "tools"
            tools.mkdir()
            calls = root / "docker-calls.jsonl"
            docker = tools / "docker"
            docker.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if sys.argv[1] == 'inspect':\n"
                "    print(os.environ['REMNANT_INTEGRATION'] + ' ' + ' '.join(['1'*40]*2))\n"
                "    raise SystemExit(0)\n"
                "with open(os.environ['DOCKER_CALLS'], 'a') as handle:\n"
                "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "raise SystemExit(77)\n"
            )
            docker.chmod(0o755)
            for name, body in (
                ("python3", "exit 0"),
                ("sleep", "exit 0"),
                ("timeout", 'shift; exec "$@"'),
                ("git", "printf '%s\\n' 1111111111111111111111111111111111111111"),
            ):
                tool = tools / name
                tool.write_text("#!/bin/sh\n" + body + "\n")
                tool.chmod(0o755)
            environment = {
                **{key: value for key, value in os.environ.items() if key not in (
                    "C4_KEEP_COUNT", "C4_LAYER_KEEP_COUNTS", "STORE_IMPLEMENTATION",
                    "REQUEST_METRICS_DIR", "SERVER_SEED", "HICACHE_RATIO",
                )},
                "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                "HOST_REPO": str(root), "LOG_HOST": str(root / "logs"),
                "DOCKER_CALLS": str(calls), "FRONTEND_DIAGNOSTICS": "0",
                "DSV4_PREPARATION_WORKERS": "1",
                "SGLANG_PY": "/fixture/selected-source/python",
            }
            script = kernels.REPO_ROOT / "scripts/local/serve.sh"
            for value in ("yes", "2", "-1", "1; exit 0"):
                with self.subTest(value=value):
                    result = subprocess.run(
                        ["bash", str(script), "native"], capture_output=True,
                        text=True, timeout=10,
                        env={**environment, "DSV4_ASYNC_PREPARATION": value},
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("DSV4_ASYNC_PREPARATION must be 0 or 1", result.stderr)
                    self.assertFalse(calls.exists())
            # Unsupported imported source must fail before kill_port's -i call.
            result = subprocess.run(
                ["bash", str(script), "native"], capture_output=True, text=True,
                timeout=10, env={**environment, "DSV4_ASYNC_PREPARATION": "1"},
            )
            self.assertEqual(result.returncode, 1)
            invocation = json.loads(calls.read_text())
            self.assertIn("SGLANG_DSV4_ASYNC_PREPARATION=1", invocation)
            self.assertNotIn("-i", invocation)
            self.assertIn("async def preflight", invocation[invocation.index("-c") + 1])
            self.assertEqual(invocation[-2:], [environment["SGLANG_PY"], "1"])
            calls.unlink()
            # Default-off does not import or activate the candidate preflight.
            result = subprocess.run(
                ["bash", str(script), "native"], capture_output=True, text=True,
                timeout=10, env={**environment, "DSV4_ASYNC_PREPARATION": "0"},
            )
            invocation = json.loads(calls.read_text())
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("-i", invocation)
            self.assertNotIn("SGLANG_DSV4_ASYNC_PREPARATION=1", invocation)

    def test_raw_jsonl_prepares_for_existing_session_replay(self):
        from scripts.local.business_replay import runner

        with tempfile.TemporaryDirectory(prefix="remnant-replay-") as temporary:
            root = Path(temporary)
            source, output = root / "raw.jsonl", root / "prepared"
            documents = [
                {"id": "later", "token_name": "session", "created": 2,
                 "messages": [{"role": "user", "content": "later"}],
                 "tools": [{"type": "function", "function": {"name": "run"}}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
                {"id": "earlier", "token_name": "session", "created": 1,
                 "messages": [{"role": "user", "content": "earlier"}]},
            ]
            lines = [json.dumps(row).encode() + b"\n" for row in documents]
            source.write_bytes(b"\n".join(lines))
            command = [sys.executable, str(kernels.REPO_ROOT / "scripts/local/convert_lswb.py"),
                       "--format", "replay", "--input", str(source), "--output-dir", str(output)]
            subprocess.run(command, cwd=root, check=True, capture_output=True, timeout=30)
            records, count = runner.load_dataset_manifest(
                output / "manifest.json", output / "records", 0, verify_snapshot=True,
            )
            self.assertEqual(count, 2)
            self.assertEqual([r.request_id for r in records], ["earlier", "later"])
            self.assertEqual({r.user_id for r in records}, {"session"})
            self.assertEqual(records[1].path.read_bytes(), lines[0])
            self.assertEqual(records[0].path.read_bytes(), lines[1])
            self.assertEqual(records[1].expected_completion_tokens, 3)
            self.assertEqual(source.read_bytes(), b"\n".join(lines))
            # The original/default bench-serving conversion remains independent.
            serving = root / "serving"
            documents[1]["usage"] = {}
            source.write_text("\n".join(json.dumps(row) for row in documents) + "\n")
            subprocess.run(
                command[:2] + ["--input", str(source), "--output-dir", str(serving)],
                cwd=root, check=True, capture_output=True, timeout=30,
            )
            rows = [json.loads(line) for line in (serving / "requests.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["messages"], documents[0]["messages"])
            self.assertEqual(rows[0]["tools"], documents[0]["tools"])
            self.assertEqual([row["max_tokens"] for row in rows], [3, 256])
            self.assertEqual(rows[0]["stream_options"], {"include_usage": True})
            self.assertNotEqual(subprocess.run(command, cwd=root, capture_output=True, timeout=30).returncode, 0)
            for bad in ({**documents[0], "created": float("nan")},
                        {**documents[0], "token_name": ""}, documents[0]):
                source.write_text(json.dumps(documents[0]) + "\n" + json.dumps(bad) + "\n")
                failed = root / "failed"
                attempt = command[:-1] + [str(failed)]
                self.assertNotEqual(subprocess.run(attempt, cwd=root, capture_output=True, timeout=30).returncode, 0)
                self.assertFalse(failed.exists())
                self.assertFalse(list(root.glob(".lswb-replay-*")))

    def test_replay_defaults_are_repo_local_and_allow_overrides(self):
        script = kernels.REPO_ROOT / "scripts/local/eval-env.sh"
        env = {k: v for k, v in os.environ.items() if not k.startswith("REPLAY_")}
        command = ["bash", "-c", '. "$1"; printf "%s\\n" "$REPLAY_RUNNER" "$REPLAY_DATASET" "$REPLAY_MANIFEST"', "bash", str(script)]
        paths = subprocess.check_output(command, cwd="/tmp", env=env, text=True, timeout=10).splitlines()
        expected = ["scripts/local/business_replay/runner.py", "data/lswb-replay/records", "data/lswb-replay/manifest.json"]
        self.assertEqual([Path(p).resolve() for p in paths], [kernels.REPO_ROOT / p for p in expected])
        env.update(REPLAY_DATASET="/custom/records", REPLAY_MANIFEST="/custom/manifest.json")
        paths = subprocess.check_output(command, env=env, text=True, timeout=10).splitlines()
        self.assertEqual(paths[1:], ["/custom/records", "/custom/manifest.json"])

    def test_lswb_summary_resolves_only_the_requested_run(self):
        with tempfile.TemporaryDirectory(prefix="remnant-row-") as directory:
            results = Path(directory) / "results' with spaces"
            base = results / "lswb-replay"
            old = base / "native-c24/20261017_120000"
            new = base / "kernel-c24/20261018_120000"
            for index, run in enumerate((old, new), 1):
                summary = run / "client/summary.json"
                summary.parent.mkdir(parents=True)
                summary.write_text(json.dumps({
                    "valid": True, "ttft_ms": {"p90": 1000},
                    "latency_ms": {"p90": 2000}, "counts": {"successful": index},
                    "success_rate": 1,
                }))
                os.utime(summary, (index, index))
            script = kernels.REPO_ROOT / "scripts/local/lswb-row.sh"

            def row(argument):
                return subprocess.run(
                    ["bash", str(script), str(argument)],
                    env={**os.environ, "RESULTS_HOST": str(results), "PYTHON": sys.executable},
                    cwd=kernels.REPO_ROOT, capture_output=True, text=True, timeout=10,
                )

            for argument in (old, old.parent, "native-c24"):
                with self.subTest(argument=str(argument)):
                    result = row(argument)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("dir=20261017_120000", result.stdout)
                    self.assertIn("done=1 ", result.stdout)
            latest = row(base)
            self.assertEqual(latest.returncode, 0, latest.stdout + latest.stderr)
            self.assertIn("dir=20261018_120000", latest.stdout)
            for argument in ("missing-tag", base / "missing-tag"):
                with self.subTest(argument=str(argument)):
                    result = row(argument)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("cannot resolve", result.stdout)
                    self.assertNotIn("slo=", result.stdout)


class ServingClientTests(unittest.TestCase):
    """Exercise the real client script; mock only external executables/server data."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="remnant-serving-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "repo with spaces"
        self.bin = self.root / "bin"
        self.bin.mkdir(parents=True)
        source = self.root / "source/sglang"
        source.mkdir(parents=True)
        (source / "__init__.py").write_text("")
        self.info = {
            "model_path": "/model with spaces", "tokenizer_path": "/tokenizer with spaces",
            "context_length": 16384, "max_running_requests": 64,
            "max_total_num_tokens": 8320, "dsv4_c4_cache_format": "native",
            "cuda_graph_config": {"decode": {"backend": "full", "bs": [1, 2, 4, 8, 16, 32]}},
        }
        self.info_path = self.root / "server-info.json"
        self.calls = self.root / "commands.jsonl"
        self.env = {
            **os.environ, "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "HOST_REPO": str(self.root), "REPO_CT": str(self.root),
            "RESULTS_HOST": str(self.root / "results"), "CONTAINER": "fixture",
            "SGLANG_PY": str(source.parent), "PYTHONPATH": str(source.parent),
            "PYTHON": str(self.bin / "python3"), "BENCH_EXECUTION": "direct",
            "BENCH_TIMEOUT_S": "5", "SEED": "42", "MOCK_INFO": str(self.info_path),
            "MOCK_CALLS": str(self.calls), "MOCK_KIND": "valid",
        }
        self.script = kernels.REPO_ROOT / "scripts/local/bench-serving.sh"
        # Fixtures emulate command contracts, never import SGLang or start a server.
        self.command("curl", """
if os.environ['MOCK_KIND'] == 'down': sys.exit(22)
if sys.argv[-1].endswith('/server_info'): print(Path(os.environ['MOCK_INFO']).read_text())
""")
        self.command("date", "print('20261018_123000')\n")
        self.command("timeout", """
import signal
args = sys.argv[1:]
while args[0].startswith('--'): args.pop(0)
seconds = float(args.pop(0).removesuffix('s'))
process = subprocess.Popen(args, start_new_session=True)
try:
    sys.exit(process.wait(timeout=seconds))
except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=2)
    sys.exit(124)
""")
        self.command("docker", """
args = sys.argv[1:]
with Path(os.environ['MOCK_CALLS']).open('a') as stream:
    stream.write(json.dumps({'docker': args}) + '\\n')
if args[0] == 'inspect': print('sha256:fixture'); sys.exit(0)
assert args.pop(0) == 'exec', args
while args[0].startswith('-'):
    option = args.pop(0)
    if option == '-e':
        key, value = args.pop(0).split('=', 1); os.environ[key] = value
    elif option == '-w': os.chdir(args.pop(0))
    else: assert option == '-i', option
assert args.pop(0) == 'fixture', args
os.execvp(args[0], args)
""")
        self.command("python3", """
if sys.argv[1:3] != ['-m', 'sglang.bench_serving']:
    os.execv(REAL_PYTHON, [REAL_PYTHON, *sys.argv[1:]])
args = sys.argv[3:]
with Path(os.environ['MOCK_CALLS']).open('a') as stream:
    stream.write(json.dumps({'bench': args}) + '\\n')
def option(name): return args[args.index(name) + 1]
destination = Path(option('--output-file'))
kind = os.environ['MOCK_KIND']
measured = destination.name == 'measured.jsonl'
if kind == 'fail' and measured: sys.exit(7)
if kind == 'timeout' and measured:
    import time
    time.sleep(10)
count = int(option('--num-prompts'))
outlen = int(option('--random-output-len'))
if kind == 'short-warmup' and not measured: count -= 1
record = {'completed': count, 'output_lens': [outlen] * count, 'errors': [''] * count}
destination.write_text(json.dumps(record) + '\\n')
if kind == 'duplicate' and measured:
    with destination.open('a') as stream: stream.write(json.dumps(record) + '\\n')
""")

    def command(self, name, body):
        path = self.bin / name
        path.write_text(
            f"#!{sys.executable}\nimport json, os, subprocess, sys\nfrom pathlib import Path\n"
            f"REAL_PYTHON = {sys.executable!r}\n" + body
        )
        path.chmod(0o755)

    def run_client(self, concurrency="4", execution="direct", kind="valid", tag="native"):
        if execution == "docker" and not (self.root / ".git").exists():
            # Exercise the real host gate as well as the mocked Docker client.
            SourcePreflightTests.initialize(self.root)
            for name in ("sglang", "flashmla"):
                fork = self.root / integration_profile(
                    self.env.get("REMNANT_INTEGRATION", "v0.5.18")
                )[name]
                SourcePreflightTests.initialize(fork)
                SourcePreflightTests.pin(self.root, fork)
        self.info_path.write_text(json.dumps(self.info))
        return subprocess.run(
            ["bash", str(self.script), tag, "30213", "1024", "16", concurrency],
            env={**self.env, "BENCH_EXECUTION": execution, "MOCK_KIND": kind},
            text=True, capture_output=True, timeout=15,
        )

    def records(self, field):
        if not self.calls.exists():
            return []
        return [r[field] for line in self.calls.read_text().splitlines()
                if field in (r := json.loads(line))]

    def test_direct_and_docker_share_workload_and_leave_server_alone(self):
        for execution in ("direct", "docker"):
            with self.subTest(execution=execution):
                result = self.run_client(execution=execution)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("server left running", result.stdout)
        waves = self.records("bench")
        self.assertEqual(len(waves), 4)
        for args, prompts in zip(waves, (4, 12, 4, 12)):
            expected = {
                "--num-prompts": str(prompts), "--max-concurrency": "4",
                "--random-input-len": "1024", "--random-output-len": "16",
                "--random-range-ratio": "1.0", "--seed": "42",
                "--warmup-requests": "0", "--request-rate": "inf",
                "--model": "/model with spaces", "--tokenizer": "/tokenizer with spaces",
            }
            for flag, value in expected.items():
                self.assertEqual(args[args.index(flag) + 1], value)
            self.assertTrue({"--flush-cache", "--tokenize-prompt", "--output-details"} <= set(args))
        self.assertTrue(all(args[0] in {"inspect", "exec"} for args in self.records("docker")))
        roots = sorted((self.root / "results/serving/native").iterdir())
        self.assertEqual(len(roots), 2)  # Identical timestamps cannot overwrite a run.
        for root in roots:
            self.assertEqual(json.loads((root / "server-info.json").read_text()), self.info)
            config = json.loads((root / "config.json").read_text())
            self.assertEqual(config["concurrency"], 4)
            self.assertEqual(config["measured_requests"], 12)
            self.assertEqual(len(config["script_sha256"]), 64)
            source = json.loads((root / "client-source.json").read_text())
            self.assertEqual(len(source["sglang_python_tree_sha256"]), 64)

    def test_auto_uses_actual_server_pool(self):
        result = self.run_client(concurrency="auto")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.records("bench")[0]
        self.assertEqual(args[args.index("--max-concurrency") + 1], "8")

    def test_local_dataset_is_forwarded_and_hashed_in_both_execution_modes(self):
        import hashlib

        dataset = self.root / "dataset with spaces.json"
        dataset.write_text(json.dumps([
            {"conversations": [{"value": "question"}, {"value": "answer"}]}
        ]))
        self.env["BENCH_DATASET_PATH"] = str(dataset)
        expected_paths = []
        for execution in ("direct", "docker"):
            with self.subTest(execution=execution):
                if execution == "docker":
                    mount = self.root.parent / "container mount"
                    mount.symlink_to(self.root, target_is_directory=True)
                    self.env["REPO_CT"] = str(mount)
                    expected_paths.extend([str(mount / dataset.name)] * 2)
                else:
                    expected_paths.extend([str(dataset)] * 2)
                result = self.run_client(execution=execution, tag=execution)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                root, = (self.root / "results/serving" / execution).iterdir()
                config = json.loads((root / "config.json").read_text())
                self.assertEqual(config["dataset"], {
                    "path": str(dataset),
                    "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                })
        for args, expected in zip(self.records("bench"), expected_paths):
            self.assertEqual(args[args.index("--dataset-path") + 1], expected)

    def test_invalid_local_datasets_fail_before_client_execution(self):
        path = self.root / "dataset.json"
        self.env["BENCH_DATASET_PATH"] = str(path)
        for payload in (None, "not JSON", "{}", "[]"):
            with self.subTest(payload=payload):
                if payload is not None:
                    path.write_text(payload)
                result = self.run_client()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.records("bench"), [])

    def test_invalid_configuration_never_starts_a_client(self):
        cases = (
            ("0", {}), ("512;exit", {}), ("4", {"context_length": 1000}),
            ("4", {"max_running_requests": 2}),
            ("auto", {"max_total_num_tokens": 1}),
            ("4", {"cuda_graph_config": {"decode": {"backend": "eager", "bs": [4]}}}),
            ("3", {"disable_cuda_graph_padding": True}),
        )
        original = copy.deepcopy(self.info)
        for concurrency, overrides in cases:
            with self.subTest(concurrency=concurrency, overrides=overrides):
                self.info = {**original, **overrides}
                result = self.run_client(concurrency=concurrency)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.records("bench"), [])

    def test_unhealthy_server_and_invalid_tag_fail_before_measurement(self):
        for kwargs in ({"kind": "down"}, {"tag": "../escape"}):
            with self.subTest(kwargs=kwargs):
                result = self.run_client(**kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.records("bench"), [])

    def test_failed_or_incomplete_waves_are_not_success(self):
        for kind in ("fail", "short-warmup", "duplicate"):
            with self.subTest(kind=kind):
                result = self.run_client(kind=kind, tag=kind)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Completed:", result.stdout)
        self.assertEqual(len(self.records("bench")), 5)

    def test_client_timeout_is_a_failure(self):
        self.env["BENCH_TIMEOUT_S"] = "1"
        result = self.run_client(kind="timeout")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("measured failed", result.stderr)


class SurvivorBudgetLauncherTests(unittest.TestCase):
    """Survivor budgets reach the existing runner without substituting modes."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="remnant-survivors-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_invalid_budgets_and_kernel_combinations_fail_before_run_context(self):
        for function in (kernels._benchmark_worker, kernels.profile_kernels):
            for count, modes in ((0, ""), (63, ""), (512, ""), (192, "kernel")):
                with (
                    self.subTest(function=function, count=count, modes=modes),
                    patch.object(kernels, "_run_context") as allocation,
                ):
                    with self.assertRaises(ValueError):
                        function.local("invalid", keep_count=count, modes=modes)
                    allocation.assert_not_called()

    def test_timing_passes_budget_modes_and_validates_the_requested_contract(self):
        contract = kernels._benchmark_cases()
        record = {}
        payload = {"fixture": "canonical timing output"}
        (self.root / "timing.json").write_text(json.dumps(payload))
        with (
            patch.object(kernels, "_benchmark_cases", return_value=contract),
            patch.object(kernels, "_run_context", return_value=nullcontext((self.root, record, float("inf")))),
            patch.object(kernels, "_run_pytest") as validity,
            patch.object(kernels, "_run_command") as command,
            patch.object(contract, "validate_timing") as check,
        ):
            self.assertEqual(
                kernels._benchmark_worker.local("count192", keep_count=192),
                str(self.root),
            )
        argv = command.call_args.args[0]
        self.assertEqual(argv[argv.index("--keep-count") + 1], "192")
        self.assertEqual(argv[argv.index("--modes") + 1], "native,triton")
        self.assertEqual(record["keep_count"], 192)
        self.assertEqual(record["modes"], ("native", "triton"))
        validity.assert_called_once()
        check.assert_called_once_with(payload, keep_count=192, modes="native,triton")

    def test_profiles_cover_only_supported_requested_modes_and_match_inputs(self):
        contract = kernels._benchmark_cases()
        record = {}
        calls = []

        def profile_command(argv, output, deadline):
            calls.append(argv)
            if capabilities := profiler_capabilities(argv):
                return capabilities
            if "--export" in argv:
                base = Path(argv[argv.index("--export") + 1])
                base.with_suffix(".ncu-rep").write_bytes(b"nonempty profile")
                base.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "status": "passed",
                            "config": {"keep_count": 192},
                            "cases": [
                                {
                                    "name": "h64_b8_k512",
                                    "valid": True,
                                    "mode": argv[argv.index("--profile-mode") + 1],
                                    "common_inputs_sha256": "identical raw inputs",
                                    "expected_kernels": ["splitkv", "combine"],
                                }
                            ],
                        }
                    )
                )
            return "exported metrics"

        with (
            patch.object(kernels, "_benchmark_cases", return_value=contract),
            patch.object(
                kernels,
                "_run_context",
                return_value=nullcontext((self.root, record, float("inf"))),
            ),
            patch.object(kernels, "_run_command", side_effect=profile_command),
            patch.object(contract, "profile_summary", return_value={}) as metrics,
            patch.object(kernels.results_volume, "commit"),
        ):
            result = kernels.profile_kernels.local(
                "count192", cases="h64_b8_k512", keep_count=192
            )
        self.assertEqual(result, str(self.root))
        captures = [argv for argv in calls if "--export" in argv]
        self.assertEqual(len(captures), 2)
        self.assertEqual({r["mode"] for r in record["runs"]}, {"native", "triton"})
        self.assertEqual(metrics.call_count, 2)
        for argv in captures:
            self.assertEqual(argv[argv.index("--keep-count") + 1], "192")
            self.assertEqual(argv[argv.index("--modes") + 1], "native,triton")
        self.assertTrue((self.root / "result.json").is_file())

    def test_profile_rejects_another_survivor_budget_or_implementation(self):
        contract = kernels._benchmark_cases()
        base = self.root / "decode-h64_b8_k512-native"
        base.with_suffix(".ncu-rep").write_bytes(b"nonempty profile")
        for count, mode in ((256, "native"), (192, "triton")):
            base.with_suffix(".json").write_text(json.dumps({
                "status": "passed", "config": {"keep_count": count},
                "cases": [{"valid": True, "mode": mode}],
            }))
            with (
                self.subTest(count=count, mode=mode),
                patch.object(kernels, "_benchmark_cases", return_value=contract),
                patch.object(kernels, "_run_context", return_value=nullcontext((self.root, {}, float("inf")))),
                patch.object(kernels, "_run_command", side_effect=profiler_capabilities),
            ):
                with self.assertRaisesRegex(RuntimeError, "Invalid capture"):
                    kernels.profile_kernels.local("count192", cases="h64_b8_k512", keep_count=192)


class PrefillReadLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="remnant-prefill-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_cpu_driver_stub_is_scoped_to_child_import_environment(self):
        cuda_home = self.root / "cuda"
        stub = cuda_home / "targets/x86_64-linux/lib/stubs/libcuda.so"
        stub.parent.mkdir(parents=True)
        stub.write_bytes(b"driver stub")
        with patch.dict(
            os.environ, {"CUDA_HOME": str(cuda_home), "LD_LIBRARY_PATH": "/real-driver"}
        ):
            environment = kernels._cpu_preflight_environment(self.root)
            self.assertEqual(os.environ["LD_LIBRARY_PATH"], "/real-driver")
            directory, original = environment["LD_LIBRARY_PATH"].split(":")
            self.assertEqual(original, "/real-driver")
            self.assertTrue(Path(directory, "libcuda.so.1").samefile(stub))
            with patch.object(kernels.subprocess, "Popen") as process:
                process.return_value.wait.return_value = 0
                deadline = kernels.time.monotonic() + 30
                kernels._run_command(
                    ["cpu-check"],
                    self.root / "cpu.log",
                    deadline,
                    environment=environment,
                )
                self.assertEqual(
                    process.call_args.kwargs["env"]["LD_LIBRARY_PATH"],
                    environment["LD_LIBRARY_PATH"],
                )
                kernels._run_command(["gpu-check"], self.root / "gpu.log", deadline)
                self.assertEqual(
                    process.call_args.kwargs["env"]["LD_LIBRARY_PATH"], "/real-driver"
                )

    def test_worker_pythonpath_excludes_serialized_launcher_shadow(self):
        shadow = self.root / "modal-bootstrap"
        shadow.mkdir()
        (shadow / "kernels.py").write_text(
            "raise RuntimeError('serialized launcher was imported')\n"
        )
        source = self.root / "python"
        source.mkdir()
        (source / "kernels.py").write_text("IDENTITY = 'intended dependency'\n")
        with (
            patch.dict(os.environ, {"PYTHONPATH": str(shadow)}),
            patch.object(kernels, "SGLANG_ROOT", self.root),
        ):
            result = kernels._run_command(
                [sys.executable, "-c", "import kernels; print(kernels.IDENTITY)"],
                self.root / "imports.log",
                kernels.time.monotonic() + 30,
            )
            self.assertEqual(result.strip(), "intended dependency")
            self.assertEqual(os.environ["PYTHONPATH"], str(shadow))

    def test_missing_cpu_driver_stub_fails_before_import(self):
        with patch.dict(os.environ, {"CUDA_HOME": str(self.root / "missing")}):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA's libcuda stub"):
                kernels._cpu_preflight_environment(self.root)

    def test_prefill_timing_uses_fixed_variants_and_coverage_judge(self):
        contract = kernels._benchmark_cases()
        payload = {"prefill": "fixture"}
        (self.root / "timing.json").write_text(json.dumps(payload))
        with (
            patch.object(kernels, "_benchmark_cases", return_value=contract),
            patch.object(
                kernels,
                "_run_context",
                return_value=nullcontext((self.root, {}, float("inf"))),
            ),
            patch.object(kernels, "_run_pytest"),
            patch.object(kernels, "_run_command") as command,
            patch.object(contract, "validate_timing") as validate,
        ):
            kernels._benchmark_worker.local("prefill", suite="prefill-read")
        argv = command.call_args.args[0]
        self.assertEqual(argv[argv.index("--suite") + 1], "prefill-read")
        self.assertNotIn("--modes", argv)
        self.assertNotIn("--keep-count", argv)
        validate.assert_called_once_with(payload, suite="prefill-read")

    def test_prefill_invalid_options_fail_before_context(self):
        for function in (kernels._benchmark_worker, kernels.profile_kernels):
            with patch.object(kernels, "_run_context") as allocation:
                with self.assertRaises(ValueError):
                    function.local("prefill", suite="prefill-read", keep_count=192)
                allocation.assert_not_called()

    def test_fused_prefill_timing_is_forwarded_and_validated_explicitly(self):
        contract = kernels._benchmark_cases()
        payload = {"candidate": "fused"}
        (self.root / "timing.json").write_text(json.dumps(payload))
        record = {}
        with (
            patch.object(kernels, "_benchmark_cases", return_value=contract),
            patch.object(
                kernels,
                "_run_context",
                return_value=nullcontext((self.root, record, float("inf"))),
            ),
            patch.object(kernels, "_run_pytest"),
            patch.object(kernels, "_run_command") as command,
            patch.object(contract, "validate_timing") as validate,
        ):
            kernels._benchmark_worker.local(
                "fused-reader", suite="prefill-read", prefill_reader="fused"
            )
        argv = command.call_args.args[0]
        self.assertEqual(argv[argv.index("--prefill-reader") + 1], "fused")
        self.assertEqual(record["prefill_reader"], "fused")
        validate.assert_called_once_with(
            payload, suite="prefill-read", prefill_reader="fused"
        )
        for function in (kernels._benchmark_worker, kernels.profile_kernels):
            for suite, reader in (("all", "fused"), ("prefill-read", "typo")):
                with patch.object(kernels, "_run_context") as context:
                    with self.assertRaisesRegex(ValueError, "prefill_reader"):
                        function.local("bad", suite=suite, prefill_reader=reader)
                    context.assert_not_called()

    def test_fused_prefill_profile_rejects_a_split_capture(self):
        contract = kernels._benchmark_cases()
        case = contract.prefill_read_cases()[0].name
        base = self.root / "prefill-read-native"
        base.with_suffix(".ncu-rep").write_bytes(b"report")
        base.with_suffix(".json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "protocol": contract.PREFILL_PROTOCOL,
                    "config": {"keep_count": 256, "prefill_reader": "split"},
                    "cases": [
                        {
                            "valid": True,
                            "name": case,
                            "capture_range": f"remnant_{case}_native",
                            "path": "native",
                            "mode": "native",
                            "keep_count": 256,
                            "path_layouts": contract.prefill_path_layouts(),
                            "prefill_reader": "split",
                        }
                    ],
                }
            )
        )
        with (
            patch.object(kernels, "_benchmark_cases", return_value=contract),
            patch.object(
                kernels,
                "_run_context",
                return_value=nullcontext((self.root, {}, float("inf"))),
            ),
            patch.object(kernels, "_run_command", side_effect=profiler_capabilities) as command,
        ):
            with self.assertRaisesRegex(RuntimeError, "Prefill capture"):
                kernels.profile_kernels.local(
                    "fused-reader",
                    cases=case,
                    suite="prefill-read",
                    prefill_reader="fused",
                )
        argv = command.call_args.args[0]
        self.assertEqual(argv[argv.index("--prefill-reader") + 1], "fused")

    def test_prefill_profile_application_replay_and_three_variant_coverage(self):
        for reader in ("split", "fused"):
            with self.subTest(prefill_reader=reader):
                self._check_prefill_profile_coverage(reader)

    def _check_prefill_profile_coverage(self, reader):
        contract = kernels._benchmark_cases()
        record = {}
        captures = []
        names = [
            c.name for c in contract.prefill_read_cases(heads=(64,), queries=(512,))[:2]
        ]
        case = ",".join(names)
        exports = []

        def run(argv, output, deadline):
            if capabilities := profiler_capabilities(argv):
                return capabilities
            if "--export" in argv:
                captures.append(argv)
                base = Path(argv[argv.index("--export") + 1])
                base.with_suffix(".ncu-rep").write_bytes(b"nonempty")
                path = argv[argv.index("--profile-path") + 1]
                base.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "status": "passed",
                            "protocol": contract.PREFILL_PROTOCOL,
                            "config": {"keep_count": 256, "prefill_reader": reader},
                            "cases": [
                                {
                                    "name": case_name,
                                    "capture_range": f"remnant_{case_name}_{path}",
                                    "valid": True,
                                    "prefill_reader": reader,
                                    "path": path,
                                    "mode": contract.prefill_path_layouts()[path][
                                        "implementation"
                                    ],
                                    "keep_count": contract.prefill_path_layouts()[path][
                                        "packed_keep_count"
                                    ]
                                    or 256,
                                    "path_layouts": contract.prefill_path_layouts(),
                                    "common_inputs_sha256": "same-" + case_name,
                                    "expected_kernels": ["dequantize"],
                                    "minimum_launches": {"dequantize": 1},
                                }
                                for case_name in argv[argv.index("--case") + 1].split(
                                    ","
                                )
                            ],
                        }
                    )
                )
            else:
                exports.append(argv)
            return "traffic"

        with (
            patch.object(kernels, "_benchmark_cases", return_value=contract),
            patch.object(
                kernels,
                "_run_context",
                return_value=nullcontext((self.root, record, float("inf"))),
            ),
            patch.object(kernels, "_run_command", side_effect=run),
            patch.object(
                contract, "profile_summary", return_value={"kernels": ["dequantize"]}
            ) as summary,
            patch.object(kernels.results_volume, "commit"),
        ):
            kernels.profile_kernels.local(
                "prefill", suite="prefill-read", cases=case, prefill_reader=reader
            )
        # The saved candidate runner batches operations per variant, retaining
        # every one of the two cases x three variants with separate exports.
        self.assertEqual(len(captures), 3)
        self.assertEqual(len(exports), 6)
        self.assertEqual(len(record["runs"]), 6)
        for argv in exports:
            self.assertEqual(argv[argv.index("--print-units") + 1], "base")
            self.assertEqual(argv[argv.index("--rename-kernels") + 1], "off")
            self.assertTrue(argv[argv.index("--nvtx-include") + 1].endswith("/"))
        self.assertEqual(
            {r["mode"] for r in record["runs"]}, set(contract.PREFILL_VARIANTS)
        )
        for argv in captures:
            self.assertEqual(argv[argv.index("--replay-mode") + 1], "application")
            self.assertIn("--profile-replay-id", argv)
            self.assertEqual(argv[argv.index("--case") + 1], case)
            self.assertIn("--nvtx", argv)
            if reader == "fused":
                self.assertEqual(argv[argv.index("--prefill-reader") + 1], reader)
            self.assertEqual(argv[argv.index("--metrics") + 1], kernels.PREFILL_METRICS)
            self.assertNotIn("--section", argv)
        self.assertTrue(all(c.kwargs["traffic_only"] for c in summary.call_args_list))
        self.assertEqual(
            {(r["case"], r["mode"]) for r in record["runs"]},
            {(name, mode) for name in names for mode in contract.PREFILL_VARIANTS},
        )




class GlmLauncherTests(unittest.TestCase):
    """GLM selection must stay explicit and fail before remote execution."""

    def test_family_mismatch_never_opens_run_context(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        with patch.object(kernels, "_run_context") as run:
            with self.assertRaisesRegex(ValueError, "model-family"):
                function("fixture", model_family="glm53")
            run.assert_not_called()

    def test_glm_and_v41_local_launcher_resolves_tp_ep_and_gpu_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            command = path / "sglang"
            command.write_text(
                "#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n"
            )
            command.chmod(0o755)
            base = {
                **os.environ,
                "PATH": str(path) + os.pathsep + os.environ["PATH"],
                "MODEL_FAMILY": "glm53",
                "REMNANT_INTEGRATION": "v0.5.22",
                "DECODE_CFG": "{}",
                "MAX_TOTAL_TOKENS": "16384",
            }
            for name in (
                "TP",
                "EP",
                "GPUS",
                "CUDA_VISIBLE_DEVICES",
                "MODEL_NAME",
                "SERVER_PID_FILE",
                "STORE_IMPLEMENTATION",
                "C4_KEEP_COUNT",
                "C4_LAYER_KEEP_COUNTS",
                "HICACHE_RATIO",
            ):
                base.pop(name, None)
            script = str(kernels.REPO_ROOT / "scripts/local/run-server.sh")
            for family in ("glm53", "dsv41"):
                for tp in (4, 8):
                    for mode in ("native", "direct"):
                        with self.subTest(tp=tp, mode=mode):
                            result = subprocess.run(
                                [
                                    "bash",
                                    script,
                                    mode,
                                    "/config-only",
                                    "127.0.0.1",
                                    "30212",
                                ],
                                env={**base, "TP": str(tp), "MODEL_FAMILY": family},
                                capture_output=True,
                                text=True,
                                check=True,
                                timeout=10,
                            )
                            args = json.loads(result.stdout)
                            for flag, expected in (
                                ("--model-type", "llm"),
                                ("--max-total-tokens", "16384"),
                                ("--tp", str(tp)),
                                ("--ep", str(tp)),
                                ("--kv-cache-dtype", "fp8_e4m3"),
                                ("--kv-cache-format", mode),
                                (
                                    "--moe-runner-backend",
                                    "deep_gemm"
                                    if family == "glm53"
                                    else "flashinfer_mxfp4",
                                ),
                            ):
                                self.assertEqual(args[args.index(flag) + 1], expected)
                            if family == "glm53":
                                self.assertEqual(
                                    args[args.index("--dsa-prefill-backend") + 1],
                                    "tilelang",
                                )
                                self.assertEqual(
                                    args[args.index("--dsa-decode-backend") + 1],
                                    "tilelang",
                                )
                            else:
                                for flag, value in (
                                    ("--dsv4-attn-backend", "flashmla"),
                                    ("--reasoning-parser", "deepseek-v41"),
                                    ("--tool-call-parser", "deepseekv41"),
                                ):
                                    self.assertEqual(args[args.index(flag) + 1], value)
                            self.assertNotIn("--dsv4-c4-cache-format", args)
                            self.assertIn("--language-model-only", args)
            for changes, error in (
                ({"TP": "4", "EP": "8"}, "positive divisor"),
                ({"TP": "8", "GPUS": "0,1,2,3"}, "visible GPUs"),
                ({"TP": "4", "GPUS": "0,1,2,"}, "visible GPUs"),
            ):
                result = subprocess.run(
                    ["bash", script, "direct", "/config-only", "127.0.0.1", "30212"],
                    env={**base, **changes},
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(error, result.stderr)

    def test_glm_container_launch_transports_named_configuration(self):
        """Execute serve.sh's real launch block with an isolated Docker stand-in.

        The stand-in retains only explicit -e settings, as docker exec does.
        No server, GPU, container or long-lived process is created.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "args.json"
            tools = root / "tools"
            tools.mkdir()
            (tools / "sglang").write_text(
                f"#!{sys.executable}\nimport json,sys\n"
                f"from pathlib import Path\np=Path({str(output)!r}); temporary=p.with_suffix('.tmp')\n"
                "temporary.write_text(json.dumps(sys.argv[1:])); temporary.replace(p)\n"
            )
            # macOS has no /proc; process-record lifecycle is tested separately.
            (tools / "setsid").write_text(
                '#!/bin/bash\nunset SERVER_PID_FILE\nexec "$@"\n'
            )
            (tools / "docker").write_text(
                f"#!{sys.executable}\nimport os,subprocess,sys,time\n"
                "args=sys.argv[1:]; assert args[:2]==['exec','-i'],args\n"
                "args=args[2:]; env={'PATH':os.environ['PATH']}\n"
                "while args and args[0]=='-e':\n"
                "    name,value=args[1].split('=',1);env[name]=value;args=args[2:]\n"
                "assert args[0]=='fixture',args\n"
                "subprocess.run(args[1:],env=env,check=True,timeout=5)\n"
                f"output={str(output)!r}\n"
                "for _ in range(100):\n"
                "    if os.path.exists(output): break\n"
                "    time.sleep(.02)\n"
                "assert os.path.exists(output),'launcher did not reach sglang'\n"
            )
            for tool in tools.iterdir():
                tool.chmod(0o755)
            serve = (kernels.REPO_ROOT / "scripts/local/serve.sh").read_text()
            start = serve.index(
                'REQUEST_METRICS_DIR="$REQUEST_METRICS_DIR_CT" ct_launch_script'
            )
            end = serve.index("\nBASH", start) + len("\nBASH")
            prefix = (
                '. "$1"\nTREE="$2"; SGLANG_PY="$2/python"; REPO_CT="$2"; MODE=direct\n'
            )
            prefix += (
                'MODEL_CT="/model with spaces"; SERVE_LOG_CT="$3"; NSYS_OUTPUT_CT=\n'
            )
            prefix += 'REQUEST_METRICS_DIR_CT="$4"\n'
            metrics = root / "container metrics with spaces"
            metrics.mkdir()
            for tp in (4, 8):
                env = {
                    **os.environ,
                    "MODEL_FAMILY": "glm53",
                "REMNANT_INTEGRATION": "v0.5.22",
                    "TP": str(tp),
                    "EP": "2",
                    "CONTAINER": "fixture",
                    "GPUS": ",".join(map(str, range(tp))),
                    "DECODE_CFG": '{"decode": {"bs": [1, 8]}}',
                    "MAX_TOTAL_TOKENS": "16384",
                    "GLM_REASONING_PARSER": "parser with spaces;$(do-not-run)",
                    "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                }
                for name in (
                    "SERVER_SEED",
                    "STORE_IMPLEMENTATION",
                    "C4_KEEP_COUNT",
                    "C4_LAYER_KEEP_COUNTS",
                    "HICACHE_RATIO",
                ):
                    env.pop(name, None)
                output.unlink(missing_ok=True)
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        prefix + serve[start:end],
                        "launch-test",
                        str(kernels.REPO_ROOT / "scripts/local/env.sh"),
                        str(kernels.REPO_ROOT),
                        str(root / "launch.log"),
                        str(metrics),
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    result.stdout + result.stderr + (root / "launch.log").read_text(),
                )
                args = json.loads(output.read_text())
                for flag, expected in (
                    ("--tp", str(tp)),
                    ("--ep", "2"),
                    ("--model-path", "/model with spaces"),
                    ("--kv-cache-format", "direct"),
                    ("--cuda-graph-config", env["DECODE_CFG"]),
                    ("--max-total-tokens", "16384"),
                    ("--reasoning-parser", env["GLM_REASONING_PARSER"]),
                    ("--export-metrics-to-file-dir", str(metrics)),
                ):
                    self.assertEqual(args[args.index(flag) + 1], expected)
                self.assertNotIn("--dsv4-c4-cache-format", args)


class GLMCPURebuildTests(unittest.TestCase):
    """The explicit rebuild excludes exactly two JIT inputs, never AOT sources."""

    def test_compile_owner_digest_excludes_only_harness_edits(self):
        from scripts.sources import glm_compile_owner_digest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            owners = [
                root / "python/sglang/benchmark/remnant/glm_artifacts.py",
                root / "python/sglang/srt/remnant/glm.py",
            ]
            test = root / "test/registered/attention/unittests/test_glm5_remnant.py"
            for path in [*owners, test]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            expected = glm_compile_owner_digest(root)
            test.write_text("unrelated fixture edits")
            self.assertEqual(glm_compile_owner_digest(root), expected)
            for path in owners:
                path.write_text("changed owner")
                self.assertNotEqual(glm_compile_owner_digest(root), expected)
                path.write_text("original")
            owners[0].unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing GLM CPU compile owner"):
                glm_compile_owner_digest(root)

    def test_legacy_image_snapshot_requires_fresh_cpu_preparation(self):
        from scripts import sources

        real_path = Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.json"
            output = root / "snapshot.json"
            import hashlib

            binary = root / "qualified.so"
            binary.write_bytes(b"qualified artifact")
            base.write_text(
                json.dumps(
                    {
                        "glm_compile": {"tilelang": []},
                        "binaries": {
                            str(binary): hashlib.sha256(binary.read_bytes()).hexdigest()
                        },
                    }
                )
            )

            def mapped(path):
                if str(path) == "/opt/remnant-build.json":
                    return base
                if str(path) == "/opt/remnant-glm-reuse-base.json":
                    return output
                return real_path(path)

            with patch.object(sources, "Path", side_effect=mapped):
                sources.snapshot_glm_artifacts(str(root))
            self.assertIsNone(json.loads(output.read_text())["compile_owner_digest"])
            owner = root / "python/sglang/benchmark/remnant/glm_artifacts.py"
            runtime = root / "python/sglang/srt/remnant/glm.py"
            for path in (owner, runtime):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("current owner")
            with patch.object(sources, "Path", side_effect=mapped):
                sources.snapshot_glm_artifacts(str(root))
            self.assertEqual(
                json.loads(output.read_text())["compile_owner_digest"],
                sources.glm_compile_owner_digest(root),
            )

            binary.write_bytes(b"changed artifact")
            with patch.object(sources, "Path", side_effect=mapped):
                with self.assertRaisesRegex(
                    RuntimeError, "Compiled base binary changed"
                ):
                    sources.snapshot_glm_artifacts(str(root))

    def test_format_transition_checks_raw_hashes_ast_and_formatter(self):
        import hashlib

        from scripts.sources import verify_glm_formatting

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python = root / "aot/python/sgl_kernel/flash_mla.py"
            cpp = root / "aot/csrc/flashmla_extension.cc"
            for path in (python, cpp):
                path.parent.mkdir(parents=True, exist_ok=True)
            old_py, new_py = b"x=1\n", b"x = 1\n"
            old_cpp, new_cpp = b"int x=1;\n", b"int x = 1;\n"
            def sha(data):
                return hashlib.sha256(data).hexdigest()
            proof = dict(
                schema="remnant-glm-format-transition-v1",
                ast_python_minor=list(sys.version_info[:2]),
                files={
                    "0/aot/python/sgl_kernel/flash_mla.py": dict(
                        before=sha(old_py),
                        after=sha(new_py),
                        ast_sha256=sha(
                            ast.dump(
                                ast.parse(old_py), include_attributes=False
                            ).encode()
                        ),
                    ),
                    "0/aot/csrc/flashmla_extension.cc": dict(
                        before=sha(old_cpp), after=sha(new_cpp)
                    ),
                },
            )
            python.write_bytes(old_py)
            cpp.write_bytes(old_cpp)
            with self.assertRaisesRegex(RuntimeError, "Python minor"):
                verify_glm_formatting(
                    {**proof, "ast_python_minor": [0, 0]}, [str(root)], candidate=True
                )

            def formatter(args, **kwargs):
                return "clang-format version 20.1.7" if "--version" in args else new_cpp

            with patch(
                "scripts.sources.subprocess.check_output", side_effect=formatter
            ):
                verify_glm_formatting(proof, [str(root)], candidate=False)
            python.write_bytes(new_py)
            cpp.write_bytes(new_cpp)
            verify_glm_formatting(proof, [str(root)], candidate=True)
            python.write_bytes(b"x = 2\n")
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                verify_glm_formatting(proof, [str(root)], candidate=True)
            proof["files"]["0/aot/python/sgl_kernel/flash_mla.py"]["after"] = sha(
                python.read_bytes()
            )
            with self.assertRaisesRegex(RuntimeError, "Python semantics"):
                verify_glm_formatting(proof, [str(root)], candidate=True)
            with self.assertRaisesRegex(RuntimeError, "Unsupported"):
                verify_glm_formatting(
                    dict(
                        schema=proof["schema"],
                        ast_python_minor=list(sys.version_info[:2]),
                        files={"0/../unsafe": {}},
                    ),
                    [str(root)],
                    candidate=True,
                )

    def test_image_source_resolver_requires_one_explicit_mode(self):
        for environment, expected in (
            ({}, (None, None)),
            ({"REMNANT_COMPILED_IMAGE": "ordinary"}, ("ordinary", None)),
            ({"REMNANT_GLM_CPU_REBUILD_FROM": "rebuild"}, (None, "rebuild")),
            ({"REMNANT_COMPILED_IMAGE": "", "REMNANT_GLM_CPU_REBUILD_FROM": "rebuild"}, (None, "rebuild")),
        ):
            with self.subTest(environment=environment):
                self.assertEqual(kernels._resolve_image_sources("glm53", environment), expected)
                self.assertEqual(kernels._resolve_image_sources("dsv4", environment), (None, None))
        conflicting = {
            "REMNANT_COMPILED_IMAGE": "inherited", "REMNANT_GLM_CPU_REBUILD_FROM": "explicit",
        }
        self.assertEqual(kernels._resolve_image_sources("dsv4", conflicting), (None, None))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            kernels._resolve_image_sources("glm53", conflicting)

    def test_conflicting_image_recipe_fails_before_source_or_image_work(self):
        with (
            patch.dict(os.environ, REMNANT_INTEGRATION="v0.5.22", REMNANT_MODEL_FAMILY="glm53",
                       REMNANT_COMPILED_IMAGE="inherited", REMNANT_GLM_CPU_REBUILD_FROM="explicit"),
            patch("scripts.sources.check_sources") as sources,
            patch.object(kernels.modal, "Image") as image,
            self.assertRaisesRegex(ValueError, "mutually exclusive"),
        ):
            runpy.run_path(str(kernels.REPO_ROOT / "scripts/modal/kernels.py"))
        sources.assert_not_called()
        self.assertEqual(image.mock_calls, [])

    def test_explicit_cpu_rebuild_never_uses_inherited_ordinary_cache(self):
        with (
            patch.dict(
                os.environ,
                REMNANT_GLM_CPU_REBUILD_FROM="explicit",
                REMNANT_COMPILED_IMAGE="inherited",
            ),
            patch.object(kernels, "_glm_launch_metadata") as metadata,
        ):
            self.assertFalse(kernels._reuse_glm_cpu_artifacts(Path("/unused"), 0, {}))
            metadata.assert_not_called()

    def test_compiled_source_allowlist(self):
        from scripts.sources import glm_rebuild_source_digest

        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / "kernels", Path(tmp) / "flash"]
            for root in roots:
                root.mkdir()
            loader = roots[0] / "ops/attention/dsa/tilelang_kernel.py"
            helper = roots[0] / "jit/include/sgl_kernel/glm_packed_decode.cuh"
            for path in (loader, helper):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("before")
            native = roots[0] / "aot.cu"
            native.write_text("native")
            flash = roots[1] / "kernel.cuh"
            flash.write_text("flash")
            expected = glm_rebuild_source_digest(roots)
            loader.write_text("after")
            helper.write_text("after")
            self.assertEqual(glm_rebuild_source_digest(roots), expected)
            native.write_text("changed")
            self.assertNotEqual(glm_rebuild_source_digest(roots), expected)
            native.write_text("native")
            flash.write_text("changed")
            self.assertNotEqual(glm_rebuild_source_digest(roots), expected)

    def test_base_identity_and_artifacts(self):
        import hashlib

        from scripts import sources

        real_path = Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = [root / "kernels", root / "flash"]
            for folder in roots:
                folder.mkdir()
            binary, cache = root / "aot.so", root / "jit.cubin"
            binary.write_bytes(b"native")
            cache.write_bytes(b"cached")
            manifest = dict(
                source_identities={
                    "sglang": {"commit": "sg"},
                    "flashmla": {"commit": "fm"},
                },
                binaries={str(binary): hashlib.sha256(binary.read_bytes()).hexdigest()},
                glm_compile={
                    "cache_files": {
                        str(cache): hashlib.sha256(cache.read_bytes()).hexdigest()
                    }
                },
            )
            base = root / "manifest.json"
            base.write_text(json.dumps(manifest))

            def mapped(path):
                if str(path) == "/opt/remnant-build.json":
                    return base
                if str(path) == "/opt/remnant-cpu-rebuild-base.json":
                    return root / "verified.json"
                return real_path(path)

            with (
                patch.object(sources, "Path", side_effect=mapped),
                patch.dict(os.environ, REMNANT_BUILT_INTEGRATION="v0.5.22"),
            ):
                digest = sources.glm_rebuild_source_digest(roots)
                expected = dict(sglang="sg", flashmla="fm")
                sources.verify_glm_cpu_rebuild(expected, digest, list(map(str, roots)))
                with self.assertRaises(RuntimeError):
                    sources.verify_glm_cpu_rebuild(
                        {**expected, "sglang": "wrong"}, digest, list(map(str, roots))
                    )
                for artifact in (binary, cache):
                    original = artifact.read_bytes()
                    artifact.write_bytes(b"changed")
                    with self.assertRaises(RuntimeError):
                        sources.verify_glm_cpu_rebuild(
                            expected, digest, list(map(str, roots))
                        )
                    artifact.write_bytes(original)
                with (
                    patch.dict(os.environ, REMNANT_BUILT_INTEGRATION="v0.5.18"),
                    self.assertRaises(RuntimeError),
                ):
                    sources.verify_glm_cpu_rebuild(
                        expected, digest, list(map(str, roots))
                    )


class GLMAttentionProtocolTests(unittest.TestCase):
    """Coverage identities and model-specific dispatch must fail closed."""

    def test_qualification_import_needs_no_historical_git_objects(self):
        import importlib

        from scripts.modal import kernel_qualification

        with patch("subprocess.check_output", side_effect=AssertionError("no Git at import")):
            policy = importlib.reload(kernel_qualification)
        self.assertEqual(len(policy.GLM_BASELINE_RUNTIME_HASHES), 5)
        self.assertTrue(all(len(value) == 64 for value in policy.GLM_BASELINE_RUNTIME_HASHES.values()))

    def test_scoped_audit_requires_helper_and_handoff_trace(self):
        for text in ("glm_packed_bytes", "glm_packed_bytes warp", "other_kernel warp CTA barrier"):
            with self.assertRaises(RuntimeError):
                kernels._check_glm_scope_audit(text, "test")
        kernels._check_glm_scope_audit("glm_packed_bytes WarpSync CTA barrier", "test")

    def test_preparation_reports_already_built_image_without_second_build(self):
        source = ast.parse((kernels.REPO_ROOT / "scripts/modal/kernels.py").read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_kernels")
        function.decorator_list = []
        image = MagicMock(object_id="im-built")
        namespace = dict(MODEL_FAMILY="glm53", kernel_image=image, SOURCE_IDENTITIES={}, json=json)
        exec(compile(ast.Module([function], type_ignores=[]), "prepare", "exec"), namespace)
        with patch("builtins.print"):
            namespace["prepare_kernels"]("glm53")
        image.build.assert_not_called()

    def test_explicit_sanitizer_selection_preserves_full_gate(self):
        self.assertEqual(kernels._selected_sanitizer_tools('racecheck', True), ('racecheck',))
        self.assertEqual(kernels._selected_sanitizer_tools('', True), kernels.SANITIZER_ORDER)
        for value, enabled in [('memcheck', False), ('memcheck,memcheck', True), ('unknown', True)]:
            with self.assertRaises(ValueError):
                kernels._selected_sanitizer_tools(value, enabled)

    def test_sanitizer_count_cannot_replace_required_case_identities(self):
        import xml.etree.ElementTree as ET

        # Construct independent fixtures instead of relying on ignored artifacts.
        names = ['test_producer_reconstruction', 'test_native_fp8_conversion',
                 'test_allocator_accounting', 'test_invalid_attention', 'test_partial_outputs',
                 'test_attention[3-8-8]', 'test_changing_input_graph[adapter]',
                 'test_changing_input_graph[direct]', 'test_production_backend_constructor',
                 'test_attention[0-1-64]', 'test_pool_lifecycle']
        root = ET.Element('testsuites')
        suite = ET.SubElement(root, 'testsuite', tests='11', errors='0', failures='0', skipped='0')
        for name in names:
            ET.SubElement(suite, 'testcase', name=name, classname='registered.attention.unittests.test_glm5_remnant')
        tree = ET.ElementTree(root)
        qualification.check_glm_sanitizer_cases(tree, 'historical', historical=True)
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            qualification.check_glm_sanitizer_cases(tree, 'current', historical=False)
        for dimensions in ((8,2,2,33),(16,4,2,66)):
            name = 'test_benchmark_sanitizer_bank_graph[' + '-'.join(map(str, dimensions)) + ']'
            node = ET.SubElement(suite, 'testcase', name=name, classname='registered.attention.unittests.test_glm5_remnant')
            geometry = dict(zip(('heads','banks','queries','forced_inner'), dimensions), warmup_calls=1, measured_warmup_calls=10, actual_warmup_calls_per_mode=dimensions[1])
            ET.SubElement(ET.SubElement(node, 'properties'), 'property', name='probe_geometry', value=json.dumps(geometry))
        suite.set('tests', '13')
        qualification.check_glm_sanitizer_cases(tree, 'current', historical=False)
        suite[0].set('name', 'unrelated_test')
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            qualification.check_glm_sanitizer_cases(tree, 'spoofed', historical=False)
        suite[0].set('name', names[0])
        ET.SubElement(suite[0], 'skipped')
        with self.assertRaises(RuntimeError):
            qualification.check_glm_sanitizer_cases(tree, 'skipped', historical=False)

    def test_glm_command_does_not_load_deepseek_native_library(self):
        with patch.object(kernels, 'MODEL_FAMILY', 'glm53'):
            command = kernels._benchmark_command('--model-family', 'glm53', '--suite', 'glm-attention')
        self.assertNotIn('--native-library', command)
        self.assertIn('glm-attention', command)

    def test_glm_sanitizer_merge_requires_clean_explicit_complete_union(self):
        import hashlib
        import xml.etree.ElementTree as ET
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(glm_compile=dict(triton=[], tilelang=[], attention_scale=0.0625), sanitizer={'binary_sha256':'tool'})
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            original = qualification.required_glm_sanitizer_cases(historical=True)
            focused = qualification.required_glm_sanitizer_cases(historical=False) - original
            def write_group(run, tool, names, interrupted=False):
                folder = root / 'microbench' / run / 'validation'
                folder.mkdir(parents=True)
                (folder / 'build-manifest.json').write_text(json.dumps(manifest))
                (folder / 'result.json').write_text(json.dumps(dict(model_family='glm53', status='failed' if interrupted else 'passed', runtime_hashes={'kernel':'same'}, sanitizer_results={name:{'status':('failed' if interrupted else 'passed') if name==tool else 'not_selected'} for name in ('memcheck','racecheck','synccheck')})))
                log = ('KeyboardInterrupt\n' if interrupted else '') + ('========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n' if tool=='racecheck' else '========= ERROR SUMMARY: 0 errors\n')
                (folder / f'{tool}.log').write_text(log)
                suite = ET.Element('testsuite', tests=str(len(names)), errors='0', failures='0', skipped='0')
                for name in sorted(names):
                    node = ET.SubElement(suite,'testcase',name=name,classname='registered.attention.unittests.test_glm5_remnant')
                    if name.startswith('test_benchmark_sanitizer_bank_graph['):
                        dimensions = tuple(map(int,name.split('[')[1][:-1].split('-')))
                        geometry = dict(zip(('heads','banks','queries','forced_inner'),dimensions), warmup_calls=1,measured_warmup_calls=10,actual_warmup_calls_per_mode=dimensions[1])
                        ET.SubElement(ET.SubElement(node,'properties'),'property',name='probe_geometry',value=json.dumps(geometry))
                if interrupted:
                    ET.SubElement(suite, 'testcase', time='0.001')  # pytest interrupted-node placeholder
                ET.ElementTree(suite).write(folder / f'{tool}.xml')
                if interrupted:
                    note = dict(schema='remnant-glm-sanitizer-checkpoint-v1',run_id=run,tool=tool,action='intentional SIGINT to owned pytest')
                    note.update({f'{suffix}_sha256':hashlib.sha256((folder / f'{tool}.{suffix}').read_bytes()).hexdigest() for suffix in ('log','xml')})
                    (folder / f'{tool}-coverage-checkpoint.json').write_text(json.dumps(note))
                return folder
            interrupted = write_group('race-original','racecheck',original,True)
            write_group('race-focused','racecheck',focused)
            write_group('sync-original','synccheck',original)
            write_group('sync-focused','synccheck',focused)
            write_group('mem-all','memcheck',original|focused)
            ids = 'race-original,race-focused,sync-original,sync-focused,mem-all'
            with patch.object(kernels,'RESULTS_ROOT',root), patch.object(kernels,'MANIFEST',manifest_path), patch.object(kernels,'_glm_benchmark_contract') as contract:
                contract.return_value.runtime_hashes.return_value = {'kernel':'same'}
                result = kernels._glm_timing_qualification(ids)
                self.assertEqual(set(result['tools']['racecheck']['cases']),original|focused)
                with self.assertRaisesRegex(RuntimeError,'missing sanitizer gates'):
                    kernels._glm_timing_qualification(ids.replace(',race-focused',''))
                exploratory = kernels._glm_timing_qualification(
                    ids.replace(',mem-all', ''), allow_incomplete=True
                )
                self.assertEqual(exploratory['status'], 'incomplete')
                self.assertEqual(set(exploratory['missing']['memcheck']), original | focused)
                self.assertFalse(exploratory['missing']['racecheck'])
                self.assertFalse(exploratory['missing']['synccheck'])
                note = interrupted / 'racecheck-coverage-checkpoint.json'
                saved = note.read_text()
                note.unlink()
                with self.assertRaisesRegex(RuntimeError,'missing sanitizer gates'):
                    kernels._glm_timing_qualification(ids)
                note.write_text(saved)
                (interrupted / 'racecheck.log').write_text('KeyboardInterrupt\n========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\nchanged')
                with self.assertRaisesRegex(RuntimeError,'report changed'):
                    kernels._glm_timing_qualification(ids)
                with self.assertRaisesRegex(RuntimeError,'report changed'):
                    kernels._glm_timing_qualification(ids, allow_incomplete=True)

    def test_glm_checkpoint_path_rejects_traversal_and_missing_records(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(kernels, 'RESULTS_ROOT', Path(directory)):
            for name in ('../other', 'bad/path', ''):
                with self.assertRaises(ValueError):
                    kernels._glm_timing_checkpoint(name)
            with self.assertRaises(FileNotFoundError):
                kernels._glm_timing_checkpoint('prior')
            expected = Path(directory) / 'microbench/prior/timing/timing.json'
            expected.parent.mkdir(parents=True)
            expected.write_text('{}')
            self.assertEqual(kernels._glm_timing_checkpoint('prior'), expected)

    def test_glm_qualified_timing_rejects_missing_tools(self):
        with self.assertRaises(ValueError):
            kernels._glm_timing_qualification('')
        with self.assertRaises(ValueError):
            kernels._glm_timing_qualification('../invalid')




class FP4QualificationTests(unittest.TestCase):
    """Fail-closed receipt aggregation, independent of GPU/native dependencies."""

    def setUp(self):
        from scripts.modal import qualification as q

        self.q = q
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = {key: {"fixture": key} for key in q.IDENTITY_FIELDS}

    def receipt(self, name, passed=(), failed=None, manifest=None, count=None, omitted_status="not_run"):
        count = self.q.FP4_CASE_COUNT if count is None else count
        root = self.root / "microbench" / name / "validation"
        root.mkdir(parents=True)
        (root / "build-manifest.json").write_text(json.dumps(manifest or self.manifest))
        states = {
            tool: {"status": "passed" if tool in passed else omitted_status}
            for tool in self.q.TOOLS
        }
        if failed:
            states[failed] = {
                "status": "failed",
                "error": "sanitizer reported a hazard",
            }
        (root / "result.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": "KeyboardInterrupt",
                    "allocation": {"status": "passed"},
                    "sanitizer_results": states,
                }
            )
        )
        for tool in passed:
            cases = "".join(
                f'<testcase classname="fp4" name="case{i}"/>' for i in range(count)
            )
            (root / f"{tool}.xml").write_text(
                f"<testsuites><testsuite>{cases}</testsuite></testsuites>"
            )

    def test_complete_tools_from_interrupted_jobs_can_combine(self):
        self.receipt("a", ("memcheck", "racecheck"))
        self.receipt("b", ("synccheck",))
        result = self.q.qualify_fp4_runs(self.root, "a,b", self.manifest)
        self.assertEqual(set(result["tools"]), self.q.TOOLS)
        self.assertEqual(result["case_count_per_tool"], self.q.FP4_CASE_COUNT)
        self.assertEqual(len(result["identity_sha256"]), 64)

    def test_unselected_tools_need_no_reports_and_clean_single_tool_runs_combine(self):
        for tool in self.q.TOOLS:
            self.receipt(tool, (tool,), omitted_status="not_selected")
        result = self.q.qualify_fp4_runs(
            self.root, ",".join(sorted(self.q.TOOLS)), self.manifest
        )
        self.assertEqual(set(result["tools"]), self.q.TOOLS)
        with self.assertRaisesRegex(RuntimeError, "incomplete/different sanitizer coverage"):
            self.q.qualify_fp4_runs(self.root, "memcheck", self.manifest)

    def partial_tool(self, name, status="interrupted", *, xml=None, log=None):
        base = self.root / "microbench" / name / "validation"
        result = base / "result.json"
        record = json.loads(result.read_text())
        record["sanitizer_results"]["racecheck"] = {"status": status}
        result.write_text(json.dumps(record))
        (base / "racecheck.xml").write_text(
            xml
            if xml is not None
            else '<testsuites><testsuite tests="1" errors="0" failures="0" skipped="0">'
            '<testcase classname="fp4" name="case0"/></testsuite></testsuites>'
        )
        (base / "racecheck.log").write_bytes(
            log if log is not None else b"========= COMPUTE-SANITIZER\n.\n"
        )
        return base

    def test_clean_partial_tools_preserve_siblings_without_adding_coverage(self):
        self.receipt("remaining", ("racecheck", "synccheck"))
        for status in ("interrupted", "not_run"):
            self.receipt(status, ("memcheck",))
            self.partial_tool(status, status)
            receipt = self.q.read_receipt(
                self.root,
                status,
                self.q.QualificationIdentity.from_manifest(self.manifest),
            )
            self.assertEqual(receipt.tools, frozenset({"memcheck"}))
            result = self.q.qualify_fp4_runs(
                self.root, f"{status},remaining", self.manifest
            )
            self.assertEqual(set(result["tools"]), self.q.TOOLS)
            self.assertEqual(result["case_count_per_tool"], self.q.FP4_CASE_COUNT)

    def test_partial_failures_cannot_be_erased_by_a_complete_later_run(self):
        self.receipt("good", self.q.TOOLS)
        failing_xml = (
            '<testsuites><testsuite tests="1" errors="0" failures="1" skipped="0">'
            '<testcase classname="fp4" name="case0"><failure message="bad output"/>'
            '</testcase></testsuite></testsuites>'
        )
        for index, (status, xml, log) in enumerate(
            (
                ("interrupted", failing_xml, None),
                ("not_run", failing_xml, None),
                (
                    "interrupted",
                    None,
                    b"========= COMPUTE-SANITIZER\n.\n"
                    b"========= Warning: Race reported between accesses\n"
                    b"========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n",
                ),
                ("interrupted", None, b"========= COMPUTE-SANITIZER\n.F\n"),
                (
                    "interrupted",
                    None,
                    b"========= COMPUTE-SANITIZER\nunknown diagnostic\n",
                ),
            )
        ):
            name = f"bad{index}"
            self.receipt(name, ("memcheck",))
            self.partial_tool(name, status, xml=xml, log=log)
            for order in (f"{name},good", f"good,{name}"):
                with (
                    self.subTest(status=status, index=index, order=order),
                    self.assertRaisesRegex(RuntimeError, "partial sanitizer"),
                ):
                    self.q.qualify_fp4_runs(self.root, order, self.manifest)

    def test_unreadable_partial_evidence_and_unfinished_states_fail_closed(self):
        self.receipt("good", self.q.TOOLS)
        for index, corruption in enumerate(
            (
                "missing_xml",
                "missing_log",
                "truncated_xml",
                "invalid_utf8",
                "running",
                "unknown",
            )
        ):
            name = f"incomplete{index}"
            self.receipt(name, ("memcheck",))
            state = corruption if corruption in {"running", "unknown"} else "interrupted"
            base = self.partial_tool(name, state)
            if corruption == "missing_xml":
                (base / "racecheck.xml").unlink()
            elif corruption == "missing_log":
                (base / "racecheck.log").unlink()
            elif corruption == "truncated_xml":
                (base / "racecheck.xml").write_text("<testsuites>")
            elif corruption == "invalid_utf8":
                (base / "racecheck.log").write_bytes(b"\xff")
            with self.subTest(corruption=corruption), self.assertRaises(RuntimeError):
                self.q.qualify_fp4_runs(self.root, f"{name},good", self.manifest)

    def test_a_later_pass_does_not_erase_a_sanitizer_failure(self):
        self.receipt("bad", ("memcheck",), failed="racecheck")
        self.receipt("good", self.q.TOOLS)
        for order in ("bad,good", "good,bad"):
            with (
                self.subTest(order=order),
                self.assertRaisesRegex(RuntimeError, "Unresolved sanitizer failure"),
            ):
                self.q.qualify_fp4_runs(self.root, order, self.manifest)

    def test_each_identity_field_is_required_and_matches(self):
        for index, key in enumerate(self.q.IDENTITY_FIELDS):
            changed = copy.deepcopy(self.manifest)
            changed[key] = "different runtime/input"
            name = f"changed{index}"
            self.receipt(name, self.q.TOOLS, manifest=changed)
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(RuntimeError, "identity changed"),
            ):
                self.q.qualify_fp4_runs(self.root, name, self.manifest)
            missing = dict(self.manifest)
            del missing[key]
            with self.assertRaisesRegex(
                RuntimeError, "Incomplete qualification identity"
            ):
                self.q.QualificationIdentity.from_manifest(missing)

    def test_missing_or_skipped_cases_never_qualify(self):
        self.receipt("short", self.q.TOOLS, count=self.q.FP4_CASE_COUNT - 1)
        with self.assertRaisesRegex(RuntimeError, "Incomplete qualification XML"):
            self.q.qualify_fp4_runs(self.root, "short", self.manifest)
        self.receipt("skipped", self.q.TOOLS)
        report = self.root / "microbench/skipped/validation/memcheck.xml"
        report.write_text(
            report.read_text().replace(
                'name="case0"/>', 'name="case0"><skipped/></testcase>'
            )
        )
        with self.assertRaisesRegex(RuntimeError, "Incomplete qualification XML"):
            self.q.qualify_fp4_runs(self.root, "skipped", self.manifest)


class FP4PreflightTests(unittest.TestCase):
    def test_sanitizer_cancellation_persists_running_then_interrupted(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        order = ("memcheck", "racecheck", "synccheck")
        record = {}
        checkpoints = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def checkpoint():
                checkpoints.append(json.loads((root / "result.json").read_text()))

            def run_pytest(tests, root, name, deadline, **kwargs):
                if name == "validation":
                    return
                persisted = json.loads((root / "result.json").read_text())
                self.assertEqual(
                    persisted["sanitizer_results"][name]["status"], "running"
                )
                if name == "racecheck":
                    (root / "racecheck.log").write_text(
                        "========= Warning: Race reported\n"
                    )
                    raise KeyboardInterrupt("budgeted cancellation")

            with (
                patch.object(kernels, "INTEGRATION", {"version": "v0.5.18"}),
                patch.object(kernels, "SANITIZER_ORDER", order),
                patch.object(
                    kernels,
                    "_run_context",
                    return_value=nullcontext((root, record, 100)),
                ),
                patch.object(kernels.time, "monotonic", return_value=0),
                patch.object(kernels, "_run_pytest", side_effect=run_pytest) as run,
                patch.object(kernels.results_volume, "commit", side_effect=checkpoint),
                self.assertRaises(KeyboardInterrupt),
            ):
                function("fixture", sanitizer=True)
            states = record["sanitizer_results"]
            self.assertEqual(states["memcheck"]["status"], "passed")
            self.assertEqual(states["racecheck"]["status"], "interrupted")
            self.assertEqual(states["synccheck"]["status"], "not_run")
            self.assertEqual(
                [c.args[2] for c in run.call_args_list],
                ["validation", "memcheck", "racecheck"],
            )
            self.assertEqual(
                [c["sanitizer_results"]["racecheck"]["status"] for c in checkpoints],
                ["not_run", "not_run", "running", "interrupted"],
            )

    def test_failed_cpu_preflight_never_schedules_an_h100(self):
        with (
            patch.object(kernels, "VALIDATION_SUITE", "v41-fp4"),
            patch.object(
                kernels._preflight_fp4_timing,
                "remote",
                side_effect=RuntimeError("bad receipt"),
            ),
            patch.object(kernels._benchmark_worker, "remote") as gpu,
        ):
            with self.assertRaisesRegex(RuntimeError, "bad receipt"):
                kernels._dispatch_benchmark(
                    "run", 256, "", "v41-fp4-decode", "split", "bad"
                )
            gpu.assert_not_called()

    def test_gpu_receives_the_successful_cpu_proof(self):
        proof = {"identity_sha256": "a" * 64, "receipts": {"clean": "b" * 64}}
        with (
            patch.object(kernels, "VALIDATION_SUITE", "v41-fp4"),
            patch.object(
                kernels._preflight_fp4_timing, "remote", return_value=proof
            ) as cpu,
            patch.object(
                kernels._benchmark_worker, "remote", return_value="result"
            ) as gpu,
        ):
            actual = kernels._dispatch_benchmark(
                "run", 256, "", "v41-fp4-decode", "split", "clean"
            )
            self.assertEqual(actual, "result")
            cpu.assert_called_once_with("run", "clean")
            bound = inspect.signature(inspect.unwrap(kernels._benchmark_worker.get_raw_f())).bind(*gpu.call_args.args, **gpu.call_args.kwargs)
            self.assertEqual(bound.arguments["preflight"], proof)



class SharedModelContractsTests(unittest.TestCase):
    def test_profiles_are_pure_and_preserve_assets(self):
        from scripts.model_profiles import model_profile

        for family in ("dsv4", "dsv41", "glm53"):
            profile = model_profile(family, "v0.5.22")
            self.assertIn(profile.revision, profile.config_url)
            self.assertEqual(len(profile.config_sha256), 64)
        self.assertEqual(
            model_profile("glm53", "v0.5.22").checkpoint_bytes, 328337455672
        )
        self.assertEqual(model_profile("glm53", "v0.5.22").volume, "glm53-flash")
        for family, version in (
            ("glm53", "v0.5.18"),
            ("dsv41", "v0.5.18"),
            ("unknown", "v0.5.22"),
        ):
            with self.assertRaises(ValueError):
                model_profile(family, version)

    def test_fp4_focused_scope_has_positive_instrumentation_and_no_full_gate(self):
        from scripts.modal.qualification import check_fp4_scope_audit

        summary = (
            "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
        )
        with self.assertRaises(RuntimeError):
            check_fp4_scope_audit(summary, "empty")
        check_fp4_scope_audit(
            summary + "flash_fwd_splitkv_mla_fp8_sparse_kernel CTA barrier", "covered"
        )
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        with (
            patch.object(kernels, "MODEL_FAMILY", "dsv4"),
            patch.object(kernels, "VALIDATION_SUITE", "v41-fp4"),
            patch.object(kernels, "_run_context") as run,
        ):
            with self.assertRaisesRegex(ValueError, "exact FP4"):
                function(
                    "wrong",
                    stage="corrective",
                    sanitizer=True,
                    sanitizer_scope="v41-merge",
                    cases=kernels.V41_MERGE_SANITIZER[0],
                )
            run.assert_not_called()
        import re
        for tool, trace in (("memcheck", False), ("synccheck", False), ("racecheck", False), ("racecheck", True)):
            prefix = kernels._fp4_sanitizer_prefix(tool, trace=trace)
            pattern = prefix[prefix.index("--kernel-name")+1].removeprefix("regex=")
            for name in ("_store", "_read", "flash_fwd_splitkv_mla_fp8_sparse_kernel",
                         "sparse_attn_fwd_kernel", "flash_fwd_mla_combine_kernel"):
                self.assertIsNotNone(re.search(pattern, name))
            self.assertIsNone(re.search(pattern, "reference_softmax_kernel"))
        with self.assertRaises(RuntimeError):
            from scripts.modal.qualification import check_sanitizer_report
            check_sanitizer_report(summary + "========= Internal Sanitizer Error: failed\n", "racecheck")
        self.assertEqual(len(kernels.V41_MERGE_SANITIZER), 15)
        self.assertEqual(len(set(kernels.V41_MERGE_SANITIZER)), 15)
        self.assertEqual(
            kernels._resolve_image_sources(
                "dsv4",
                {
                    "REMNANT_VALIDATION_SUITE": "v41-fp4",
                    "REMNANT_COMPILED_IMAGE": "verified",
                },
            ),
            ("verified", None),
        )
        with self.assertRaisesRegex(ValueError, "requires glm53"):
            kernels._resolve_image_sources(
                "dsv4",
                {
                    "REMNANT_VALIDATION_SUITE": "v41-fp4",
                    "REMNANT_GLM_CPU_REBUILD_FROM": "wrong",
                },
            )

    def test_parallel_smoke_validates_before_scheduling(self):
        from scripts.modal import serve

        for family, mode in (
            ("dsv4", "native"),
            ("glm53", "adapter"),
            ("dsv41", "packed"),
        ):
            with (
                patch.object(serve, "MODEL_FAMILY", family),
                patch.object(serve.prepare_serving_smoke, "remote") as prepare,
                patch.object(serve._smoke_serving_worker, "remote") as worker,
            ):
                with self.assertRaises(ValueError):
                    serve.smoke_serving("fixture", mode)
                prepare.assert_not_called()
                worker.assert_not_called()

    def test_v41_resolved_smoke_settings_require_backend_and_tuning(self):
        from scripts.modal import serve

        settings = dict(
            kv_cache_format="direct",
            tp_size=4,
            ep_size=4,
            kv_cache_dtype="fp8_e4m3",
            language_model_only=True,
            context_length=8192,
            max_running_requests=4,
            chunked_prefill_size=256,
            max_total_num_tokens=16384,
            dsv4_attn_backend="flashmla",
            disable_flashinfer_autotune=True,
            reasoning_parser="deepseek-v41",
            tool_call_parser="deepseekv41",
            cuda_graph_config={"decode": {"backend": "full", "bs": [1, 2, 4]}},
        )
        for change, succeeds in (
            ({}, True),
            ({"kv_cache_format": "native"}, False),
            ({"dsv4_attn_backend": "auto"}, False),
            ({"disable_flashinfer_autotune": False}, False),
        ):
            with patch.object(
                serve.urllib.request,
                "urlopen",
                return_value=io.BytesIO(json.dumps({**settings, **change}).encode()),
            ):
                if succeeds:
                    self.assertEqual(
                        serve._smoke_settings(
                            "30211", "direct", serve.time.monotonic() + 10, "dsv41"
                        ),
                        settings,
                    )
                else:
                    with self.assertRaises(RuntimeError):
                        serve._smoke_settings(
                            "30211", "direct", serve.time.monotonic() + 10, "dsv41"
                        )

    def test_single_mode_smoke_owns_only_requested_mode_and_result_directory(self):
        from contextlib import contextmanager

        from scripts.modal import serve

        with tempfile.TemporaryDirectory() as tmp:
            for family in ("glm53", "dsv41"):
                for mode in ("native", "direct"):
                    exited = []

                    @contextmanager
                    def boot(environment, log, deadline):
                        self.assertEqual(
                            float(environment["MEM_FRAC"]),
                            {"glm53": 0.86, "dsv41": 0.90}[family],
                        )
                        log.write_text(
                            "\n".join(
                                f"[2026-11-05 12:00:00 TP{rank} EP{rank}] {marker}"
                                for rank in range(4)
                                for marker in (
                                    "Load weight end.",
                                    "Capture target decode CUDA graph end.",
                                )
                            )
                        )
                        try:
                            yield
                        finally:
                            exited.append(True)

                    with (
                        patch.object(serve, "MODEL_FAMILY", family),
                        patch.object(serve, "RESULTS_ROOT", Path(tmp)),
                        patch.object(
                            serve, "_check_serving_smoke_image", return_value={}
                        ),
                        patch.object(
                            serve, "_check_serving_checkpoint", return_value={}
                        ),
                        patch.object(
                            serve.subprocess,
                            "check_output",
                            return_value="NVIDIA H200\n" * 4,
                        ),
                        patch.object(
                            serve, "_running_serving_server", side_effect=boot
                        ),
                        patch.object(serve, "_smoke_settings", return_value={}),
                        patch.object(serve, "_smoke_chat_request", return_value={}),
                        patch.object(serve.results_volume, "commit"),
                        patch("builtins.print"),
                    ):
                        result = serve._run_serving_smoke(
                            "shared-pair",
                            mode,
                            allocation_seconds=serve._serving_smoke_profile(
                                family
                            ).mode_seconds
                            + 60,
                            artifact_mode=mode,
                        )
                        self.assertEqual(set(result["mode_results"]), {mode})
                        self.assertEqual(
                            result["mode_results"][mode]["status"], "passed"
                        )
                        self.assertEqual(exited, [True])
                for mode in ("native", "direct"):
                    artifact = (
                        Path(tmp)
                        / (family + "-serving")
                        / "shared-pair"
                        / mode
                        / "result.json"
                    )
                    self.assertEqual(
                        set(json.loads(artifact.read_text())["mode_results"]), {mode}
                    )

    def test_shared_report_parsing_and_new_raw_identity(self):
        import hashlib

        from scripts.modal.qualification import (read_pytest_report,
                                                 validate_run_id,
                                                 verify_sanitizer_artifacts)

        for name in ("../bad", "a/b", "", "a" * 129):
            with self.assertRaises(ValueError):
                validate_run_id(name)
        validate_run_id("good-run_1")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "memcheck.xml").write_text(
                '<testsuite tests="1"><testcase classname="unit" name="case"><properties><property name="shape" value="8"/></properties></testcase></testsuite>'
            )
            (root / "memcheck.log").write_text("========= ERROR SUMMARY: 0 errors\n")
            cases, counts = read_pytest_report(root / "memcheck.xml")
            self.assertEqual(counts, dict(tests=1, errors=0, failures=0, skipped=0))
            self.assertEqual(cases[0].find("properties/property").get("value"), "8")
            state = {
                "report_schema": "remnant-sanitizer-raw-v1",
                **{
                    f"{s}_sha256": hashlib.sha256(
                        (root / f"memcheck.{s}").read_bytes()
                    ).hexdigest()
                    for s in ("log", "xml")
                },
            }
            verify_sanitizer_artifacts(root, "memcheck", state)
            (root / "memcheck.log").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "raw report changed"):
                verify_sanitizer_artifacts(root, "memcheck", state)

    def test_glm_smoke_requires_all_rank_progress_and_resolved_mode(self):
        from scripts.modal import serve

        rows = [
            f"[2026-11-04 13:53:46 TP{rank} EP{rank}] {marker}"
            for rank in range(4)
            for marker in ("Load weight end.", "Capture target decode CUDA graph end.")
        ]
        stages = serve._glm_smoke_stages("\n".join(rows))
        for event in ("weight_load_complete", "graph_capture_complete"):
            self.assertEqual(
                {s["rank"] for s in stages if s["event"] == event}, set(range(4))
            )
        settings = dict(
            kv_cache_format="direct",
            tp_size=4,
            ep_size=4,
            kv_cache_dtype="fp8_e4m3",
            dsa_prefill_backend="tilelang",
            dsa_decode_backend="tilelang",
            language_model_only=True,
            context_length=8192,
            max_running_requests=4,
            chunked_prefill_size=2048,
            max_total_num_tokens=16384,
            cuda_graph_config={"decode": {"backend": "full", "bs": [1, 2, 4]}},
        )
        response = io.BytesIO(json.dumps(settings).encode())
        with patch.object(serve.urllib.request, "urlopen", return_value=response):
            self.assertEqual(
                serve._glm_smoke_settings(
                    "30211", "direct", serve.time.monotonic() + 10
                ),
                settings,
            )
        response = io.BytesIO(json.dumps(settings).encode())
        with (
            patch.object(serve.urllib.request, "urlopen", return_value=response),
            self.assertRaisesRegex(RuntimeError, "kv_cache_format"),
        ):
            serve._glm_smoke_settings("30211", "native", serve.time.monotonic() + 10)

    def test_ambiguous_sanitizers_fail_before_run_context(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        with patch.object(kernels, "_run_context") as context:
            with self.assertRaises(ValueError):
                function(
                    "ambiguous",
                    sanitizer=True,
                    sanitizer_tools="memcheck",
                    sanitizer_order="memcheck,racecheck,synccheck",
                )
            context.assert_not_called()

    def test_merge_sanitizer_scope_and_fp4_allowlist_fail_before_execution(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        with (
            patch.object(kernels, "MODEL_FAMILY", "glm53"),
            patch.object(kernels, "_run_context") as context,
        ):
            with self.assertRaisesRegex(ValueError, "exact loader/decode/prefill"):
                function(
                    "merge-scope",
                    model_family="glm53",
                    stage="corrective",
                    sanitizer=True,
                    sanitizer_scope="glm-merge",
                    cases=kernels.GLM_MERGE_SANITIZER[0],
                )
            context.assert_not_called()
        with (
            patch.object(kernels, "MODEL_FAMILY", "dsv4"),
            patch.object(kernels, "VALIDATION_SUITE", "v41-fp4"),
            patch.object(kernels, "_run_context") as context,
        ):
            with self.assertRaisesRegex(ValueError, "allowlisted"):
                function(
                    "fp4-case",
                    stage="corrective",
                    cases=kernels.TEST_ROOT + "test_remnant_fp4.py::test_not_owned",
                )
            context.assert_not_called()

    def test_cpu_preflight_does_not_commit_gpu_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            with (
                patch.object(kernels, "MANIFEST", manifest),
                patch.object(kernels, "RESULTS_ROOT", root),
                patch.object(kernels, "results_volume"),
                patch.object(kernels, "triton_volume") as cache,
            ):
                with kernels._run_context("cpu", "preflight", 1, resource="cpu"):
                    pass
                cache.commit.assert_not_called()


class LswbSweepTests(unittest.TestCase):
    """Author: Winston Cai. Replay and server boundaries are CPU-only fakes."""

    def setUp(self):
        from scripts.local.business_replay import sweep
        self.sweep = sweep
        temporary = tempfile.TemporaryDirectory(prefix="lswb-sweep-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = "sha256:" + "1" * 64
        self.env = {
            "HOST_REPO": str(self.root), "RESULTS_HOST": str(self.root / "results"),
            "CONTAINER": "fixture", "PORT": "30381", "MASTER_PORT": "29781", "TP": "4",
            "MEM_FRAC": ".88", "MAX_RUN": "256", "DECODE_CFG": json.dumps({"decode": {"backend": "full", "bs": [4, 8, 32]}}),
            "CTX_LEN": "1048576", "CHUNK": "8192", "MODEL_CT": "/model", "MODEL_NAME": "fixture",
            "REPLAY_RUNNER": str(self.root / "runner.py"), "REPLAY_ADAPTER": str(self.root / "adapter.py"),
            "REPLAY_MANIFEST": str(self.root / "dataset.json"), "REPLAY_DATASET": str(self.root / "data"),
            "CUDA_VISIBLE_DEVICES": "4,5,6,7", "GPUS": "4,5,6,7", "REMNANT_INTEGRATION": "v0.5.18",
        }
        self.sources = {"parent": {"commit": "parent"}, "sglang": {"commit": "fork", "dirty": False}}
        self.info = {
            "model_path": "/model", "served_model_name": "fixture", "export_metrics_to_file": True, "enable_metrics": True,
            "tp_size": 4, "mem_fraction_static": .88, "random_seed": 20260919,
            "chunked_prefill_size": 8192, "context_length": 1048576, "max_running_requests": 256,
            "enable_hierarchical_cache": False, "dsv4_c4_cache_format": "remnant",
            "dsv4_c4_cache_implementation": "kernel", "dsv4_c4_cache_store_implementation": "kernel",
            "dsv4_c4_cache_keep_count": 256, "dsv4_c4_cache_layer_keep_counts": None,
            "cuda_graph_config": json.loads(self.env["DECODE_CFG"]), "max_total_num_tokens": 4536832,
        }

    def args(self, *extra):
        return self.sweep.parse_args(["direct", "--concurrencies", "2,3", "--tag", "screen",
                                      "--expected-image-id", self.image, *extra])

    def manifest(self, args):
        return {"points": [{"concurrency": c, "repeat": r} for c in args.concurrencies
                           for r in range(1, args.repeats + 1)],
                "settings": dict(self.env), "protocol": self.sweep.runner.PROTOCOL,
                "duration_s": args.duration_s, "minimum_success_rate": args.minimum_success_rate,
                "ttft_slo_ms": args.ttft_slo_ms, "source_hashes": {self.env["REPLAY_MANIFEST"]: "dataset-hash"}}

    def fixture(self, path, point, manifest, *, fifo=False):
        """Independent tiny raw workload, summarized by the existing replay owner."""
        runner = self.sweep.runner
        path.mkdir(parents=True)
        (path / "client").mkdir()
        records = [runner.Record(str(i), "user", 1, Path(f"{i}.json"), "sha", 1, 2) for i in range(2)]
        clients = [dict(request_id="0", user_id="user", valid=True, errors=[], prompt_tokens=1,
                        completion_tokens=2, ttft_ms=10000., latency_ms=12000., tpot_ms=2000.,
                        cached_tokens_details={"device": 0, "host": 0, "storage": 0},
                        cached_tokens_details_status="requested_zero_hit", completed_offset_s=12.,
                        dispatch_offset_s=0., ready_ttft_ms=10000., ready_latency_ms=12000.),
                   dict(request_id="1", user_id="user", valid=False, errors=["missing_output_event"],
                        completed_offset_s=13., dispatch_offset_s=12.)]
        activity = [{"event": "user_activate", "user_id": "user", "active": 1, "active_users": 1},
                    {"event": "ready", "request_id": "0", "user_sequence": 0,
                     "ready_offset_s": 0., "session_slot_id": 0},
                    {"event": "ready", "request_id": "1", "user_sequence": 1,
                     "ready_offset_s": 12., "session_slot_id": 0}]
        if fifo:
            slots = runner.request_slots
            outcome = slots.ReplayOutcome(manifest["duration_s"], 13., activity=activity)
            outcome.states = [slots.RequestState(r, i, float(i * 12), 0, float(i * 12), c)
                              for i, (r, c) in enumerate(zip(records, clients))]
            summary = slots.summarize(outcome, records, manifest["minimum_success_rate"],
                                      manifest["ttft_slo_ms"], runner.summarize, runner.distribution)
        else:
            summary = runner.summarize(records, clients, activity, manifest["duration_s"], manifest["minimum_success_rate"])
        record_rows = [{**r.__dict__, "path": str(r.path)} for r in records]
        for relative, data in {
            "client/summary.json": summary,
            "client/done.json": {"valid": summary["valid"], "protocol": manifest["protocol"]},
            "request-timing-validation.json": {"complete": True, "partial": False, "errors": [], "records": 2},
            "replay-window.json": {"started_at": 100., "ended_at": 100. + manifest["duration_s"]},
            "client/dataset_manifest.json": {"records": record_rows, "source_manifest_sha256": "dataset-hash"},
            "client/manifest.json": {"replay": {"max_concurrency": point["concurrency"], "max_duration_s": manifest["duration_s"]}},
        }.items():
            (path / relative).write_text(json.dumps(data))
        for relative, data in (("client/responses.jsonl", clients), ("client/active_requests.jsonl", activity)):
            (path / relative).write_text("".join(json.dumps(row) + "\n" for row in data))
        (path / "cache-flush.log").write_text("Cache flushed.\n")
        (path / "request-timings.jsonl").write_text('{"id":"server-0"}\n{"id":"server-1"}\n')
        for relative in ("client.log", "server.log"):
            (path / relative).write_text("fixture\n")
        (path / "server-info.json").write_text(json.dumps(self.info))
        return 0 if summary["valid"] else 1

    def test_defaults_and_invalid_arguments(self):
        args = self.args()
        self.assertEqual((args.duration_s, args.repeats, args.point_timeout_s), (1200, 1, 2100))
        for extra in (("--concurrencies", "2,2"), ("--concurrencies", "2,,3"),
                      ("--duration-s", "0"), ("--repeats", "-1"), ("--tag", "../bad"),
                      ("--expected-image-id", "latest"), ("--ttft-slo-ms", "nan"),
                      ("--minimum-success-rate", "1.1"), ("--point-timeout-s", "1200")):
            with self.subTest(extra=extra), patch("sys.stderr"), self.assertRaises(SystemExit):
                self.args(*extra)

    def test_help_requires_no_docker_or_data(self):
        script = kernels.REPO_ROOT / "scripts/local/sweep-lswb.sh"
        result = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True,
                                env={"PATH": os.environ["PATH"], "PYTHON": sys.executable}, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--expected-image-id", result.stdout)

    def prepare_mocks(self, args, env=None, image=None):
        env = env or self.env
        container = [{"Image": image or self.image, "State": {"Running": True}}]
        return patch.multiple(self.sweep, helper=MagicMock(return_value=""),
                              source_identity=MagicMock(return_value=self.sources),
                              source_hashes=MagicMock(return_value={}), check_ports=MagicMock(),
                              check_owner_record=MagicMock(),
                              command=MagicMock(return_value=subprocess.CompletedProcess([], 0, json.dumps(container))))

    def test_preflight_freezes_order_and_defaults(self):
        args = self.args("--repeats", "3")
        with self.prepare_mocks(args), patch.object(self.sweep.runner, "load_dataset_manifest", return_value=([], 0)):
            manifest, env = self.sweep.prepare(args, self.env)
        self.assertEqual(manifest["points"], [dict(concurrency=c, repeat=r) for c in (2, 3) for r in (1, 2, 3)])
        self.assertEqual(env["GPUS"], "4,5,6,7")
        self.assertEqual(env["SERVER_SEED"], "20260919")
        self.assertNotIn("DSV4_ASYNC_PREPARATION", env)
        self.assertNotIn("HICACHE_RATIO", env)

    def test_image_mismatch_and_source_failure_prevent_startup(self):
        args = self.args()
        with self.prepare_mocks(args, image="sha256:" + "2" * 64), self.assertRaisesRegex(ValueError, "image mismatch"):
            self.sweep.prepare(args, self.env)
        with self.prepare_mocks(args):
            with patch.object(self.sweep, "source_identity", side_effect=RuntimeError("dirty fork")), self.assertRaisesRegex(RuntimeError, "dirty fork"):
                self.sweep.prepare(args, self.env)

    def test_gpuq_assignment_required_except_dry_run(self):
        env = {k: v for k, v in self.env.items() if k != "CUDA_VISIBLE_DEVICES"}
        with self.assertRaisesRegex(ValueError, "GPUQ"):
            self.sweep.prepare(self.args(), env)
        args = self.args("--dry-run")
        with self.prepare_mocks(args), patch.object(self.sweep.runner, "load_dataset_manifest", return_value=([], 0)):
            _, resolved = self.sweep.prepare(args, env)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", resolved)

    def test_occupied_port_and_ownership_record(self):
        import socket
        with socket.socket() as occupied:
            occupied.bind(("0.0.0.0", 0))
            env = {**self.env, "PORT": str(occupied.getsockname()[1])}
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                self.sweep.check_ports(env)
        with patch.object(self.sweep, "command", return_value=subprocess.CompletedProcess([], 0, "")), self.assertRaisesRegex(RuntimeError, "ownership record"):
            self.sweep.check_owner_record(self.env)

    def test_hicache_requires_bound_reset_evidence(self):
        env = {**self.env, "HICACHE_RATIO": "2", "SWEEP_MODE": "direct"}
        with self.assertRaisesRegex(ValueError, "HICACHE_RESET_RECEIPT"):
            self.sweep.validate_hicache_receipt(env, self.image, self.sources)
        evidence = self.root / "reset.json"
        evidence.write_text("GPU reset observation")
        receipt = {"image_id": self.image, "integration": "v0.5.18", "mode": "direct", "hicache_ratio": 2.,
                   "fork_commits": {"sglang": "fork"}, "checks": {key: True for key in (
                       "gpu_cache_empty", "host_cache_empty", "transfers_drained")},
                   "evidence_path": str(evidence), "evidence_sha256": self.sweep.sha256(evidence)}
        path = self.root / "receipt.json"
        path.write_text(json.dumps(receipt))
        env["HICACHE_RESET_RECEIPT"] = str(path)
        self.assertIsNotNone(self.sweep.validate_hicache_receipt(env, self.image, self.sources))
        evidence.write_text("changed")
        with self.assertRaisesRegex(ValueError, "matching SHA256"):
            self.sweep.validate_hicache_receipt(env, self.image, self.sources)

    def test_gate_boundaries_and_recomputed_values(self):
        args = self.args("--minimum-success-rate", "0.5")
        manifest, point = self.manifest(args), dict(concurrency=2, repeat=1)
        path = self.root / "point"
        code = self.fixture(path, point, manifest)
        row = self.sweep.collect_point(path, point, manifest, code)
        self.assertTrue(row["measurement_complete"])
        self.assertTrue(row["response_success_pass"])
        self.assertFalse(row["ttft_slo_pass"])  # Exactly 10,000 ms is a failure.
        self.assertEqual(row["request_throughput"], 1 / 1200)
        self.assertEqual(row["completion_token_throughput"], 2 / 1200)
        path_summary = path / "client/summary.json"
        summary = json.loads(path_summary.read_text())
        summary["request_throughput"] += 1
        path_summary.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "raw records"):
            self.sweep.collect_point(path, point, manifest, code)

    def test_fifo_uses_existing_accounting_and_ready_slo(self):
        args = self.args()
        manifest = self.manifest(args)
        manifest["protocol"] = self.sweep.runner.request_slots.PROTOCOL
        point, path = dict(concurrency=2, repeat=1), self.root / "fifo"
        code = self.fixture(path, point, manifest, fifo=True)
        row = self.sweep.collect_point(path, point, manifest, code)
        self.assertFalse(row["response_success_pass"])
        self.assertFalse(row["ttft_slo_pass"])

    def run_sweep(self, failure=None):
        args = self.args("--repeats", "2")
        manifest = self.manifest(args)
        env = {**self.env, "SERVER_SEED": "20260919"}
        calls = []
        def fake_command(arguments, environment, **kwargs):
            calls.append(arguments)
            if "bench-lswb.sh" in str(arguments):
                path = Path(environment["LSWB_RUN_ROOT"])
                if failure:
                    raise failure
                point = {"concurrency": int(arguments[-2]), "repeat": int(path.name.rsplit("-", 1)[1])}
                code = self.fixture(path, point, manifest)
                return subprocess.CompletedProcess(arguments, code, "")
            return subprocess.CompletedProcess(arguments, 0, "gpu0,H100\n")
        with (patch.object(self.sweep, "command", side_effect=fake_command),
              patch.object(self.sweep, "check_ports"), patch.object(self.sweep, "check_owner_record"),
              patch.object(self.sweep, "server_info", return_value=self.info),
              patch.object(self.sweep, "check_frozen"), patch("builtins.print")):
            if failure:
                with self.assertRaisesRegex(RuntimeError, "Sweep failed"):
                    self.sweep.execute(args, manifest, env)
                root = next((Path(env["RESULTS_HOST"]) / "lswb-replay").iterdir())
            else:
                root = self.sweep.execute(args, manifest, env)
        return root, calls

    def test_one_boot_cleanup_repeat_order_and_continue_measured_failures(self):
        root, calls = self.run_sweep()
        serve = [c for c in calls if "serve.sh" in str(c)]
        self.assertEqual([c[-1] for c in serve], ["direct", "stop"])
        rows = json.loads((root / "summary.json").read_text())["points"]
        self.assertEqual([(r["concurrency"], r["repeat"]) for r in rows], [(2, 1), (2, 2), (3, 1), (3, 2)])
        self.assertTrue(all(r["measurement_complete"] for r in rows))
        self.assertTrue(all(not r["response_success_pass"] and not r["ttft_slo_pass"] for r in rows))
        self.assertEqual(len({r["artifact_path"] for r in rows}), 4)
        self.assertEqual(json.loads((root / "summary.json").read_text())["status"], "complete")

    def test_timeout_and_health_failure_stop_and_cleanup(self):
        root, calls = self.run_sweep(subprocess.TimeoutExpired("replay", 2100))
        self.assertEqual(sum("bench-lswb.sh" in str(c) for c in calls), 1)
        self.assertEqual(calls[-1][-1], "stop")
        self.assertEqual(json.loads((root / "summary.json").read_text())["status"], "failed")

    def test_interruption_stops_remaining_points_and_cleans_up(self):
        root, calls = self.run_sweep(InterruptedError("interrupt"))
        self.assertEqual(sum("bench-lswb.sh" in str(c) for c in calls), 1)
        self.assertEqual(calls[-1][-1], "stop")
        self.assertEqual(json.loads((root / "summary.json").read_text())["status"], "failed")

    def test_local_command_timeout_terminates_its_process_group(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.sweep.command([sys.executable, "-c", "import time; time.sleep(60)"],
                               self.env, timeout=1)

    def test_startup_failure_and_missing_artifacts_cleanup(self):
        args = self.args()
        for boot_fails in (True, False):
            with self.subTest(boot_fails=boot_fails):
                calls = []
                def fake(arguments, env, **kwargs):
                    calls.append(arguments)
                    if boot_fails and str(arguments[-1]) == "direct":
                        raise RuntimeError("startup failed")
                    return subprocess.CompletedProcess(arguments, 0, "")
                with (patch.object(self.sweep, "command", side_effect=fake),
                      patch.object(self.sweep, "check_ports"), patch.object(self.sweep, "check_owner_record"),
                      patch.object(self.sweep, "check_frozen"), patch.object(self.sweep, "server_info", return_value=self.info),
                      patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "Sweep failed")):
                    self.sweep.execute(args, self.manifest(args), {**self.env, "SERVER_SEED": "20260919"})
                self.assertEqual(calls[-1][-1], "stop")

    def test_changed_server_or_source_is_operational_failure(self):
        with patch.object(self.sweep, "source_identity", return_value={"sglang": {"commit": "changed"}}), self.assertRaisesRegex(RuntimeError, "Frozen source"):
            self.sweep.check_frozen({"sources": self.sources, "source_hashes": {}}, self.env)
        for change in ({"tp_size": 2}, {"enable_hierarchical_cache": True}, {"dsv4_c4_cache_implementation": "triton"}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Server configuration mismatch"):
                self.sweep.check_server({**self.info, **change}, {**self.env, "SERVER_SEED": "20260919"}, "direct", 3)

    def test_shared_launch_configuration_changes_invalidate_source_hashes(self):
        with patch.object(self.sweep, "sha256", return_value="probe"):
            paths = self.sweep.source_hashes(self.env)
        helper = self.root / "scripts/local/launch-config.sh"
        self.assertIn(str(helper.resolve()), paths)
        for name in paths:
            path = Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("baseline")
        baseline = self.sweep.source_hashes(self.env)
        helper.write_text("changed launch settings")
        self.assertNotEqual(self.sweep.source_hashes(self.env), baseline)

    def test_unsupported_models_fail_before_preflight_or_startup(self):
        for family in ("glm53", "dsv41"):
            with self.subTest(family=family), patch.object(self.sweep, "command") as command:
                with self.assertRaisesRegex(ValueError, "MODEL_FAMILY=dsv4 only"):
                    self.sweep.prepare(self.args(), {**self.env, "MODEL_FAMILY": family})
                command.assert_not_called()

    def test_generic_selector_is_verified_for_both_v0522_modes(self):
        env = {**self.env, "REMNANT_INTEGRATION": "v0.5.22", "SERVER_SEED": "20260919"}
        info = {key: value for key, value in self.info.items()
                if key not in {"dsv4_c4_cache_format", "dsv4_c4_cache_implementation"}}
        for mode in ("native", "direct"):
            with self.subTest(mode=mode):
                resolved = self.sweep.check_server({**info, "kv_cache_format": mode}, env, mode, 3)
                self.assertEqual(resolved["kv_cache_format"], mode)
                with self.assertRaisesRegex(ValueError, "kv_cache_format"):
                    self.sweep.check_server({**info, "kv_cache_format": "wrong"}, env, mode, 3)

    def test_dry_run_does_not_execute_or_create_results(self):
        args = self.args("--dry-run")
        with (patch.object(self.sweep, "parse_args", return_value=args),
              patch.object(self.sweep, "prepare", return_value=({}, self.env)),
              patch.object(self.sweep, "execute") as execute, patch("builtins.print")):
            self.assertEqual(self.sweep.main(), 0)
        execute.assert_not_called()
        self.assertFalse(Path(self.env["RESULTS_HOST"]).exists())

    def test_health_failure_stops_before_first_replay_and_cleans_up(self):
        args = self.args()
        calls = []
        def fake(arguments, env, **kwargs):
            calls.append(arguments)
            return subprocess.CompletedProcess(arguments, 0, "gpu,H100\n")
        with (patch.object(self.sweep, "command", side_effect=fake),
              patch.object(self.sweep, "check_ports"), patch.object(self.sweep, "check_owner_record"),
              patch.object(self.sweep, "server_info", side_effect=OSError("unhealthy")),
              patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "unhealthy")):
            self.sweep.execute(args, self.manifest(args), {**self.env, "SERVER_SEED": "20260919"})
        self.assertEqual(calls[-1][-1], "stop")
        self.assertFalse(any("bench-lswb.sh" in str(c) for c in calls))

    def test_incomplete_and_inconsistent_artifacts_are_rejected(self):
        args = self.args()
        manifest, point = self.manifest(args), dict(concurrency=2, repeat=1)
        for issue in ("timing", "missing", "exit", "dataset"):
            with self.subTest(issue=issue):
                path = self.root / issue
                code = self.fixture(path, point, manifest)
                if issue == "timing":
                    (path / "request-timing-validation.json").write_text('{"complete":false}')
                elif issue == "missing":
                    (path / "request-timings.jsonl").unlink()
                elif issue == "dataset":
                    target = path / "client/dataset_manifest.json"
                    data = json.loads(target.read_text())
                    data["source_manifest_sha256"] = "changed"
                    target.write_text(json.dumps(data))
                else:
                    code = 0
                with self.assertRaises(ValueError):
                    self.sweep.collect_point(path, point, manifest, code)

    def test_standalone_replay_output_and_response_threshold(self):
        """Run the real replay shell with a fake HTTP endpoint and client, no GPUs."""
        repo = self.root / "standalone"
        local = repo / "scripts/local"
        local.mkdir(parents=True)
        (local / "bench-lswb.sh").write_text((kernels.REPO_ROOT / "scripts/local/bench-lswb.sh").read_text())
        fake_runner = repo / "client.py"
        fake_runner.write_text(
            "import json, pathlib, sys\n"
            "args=sys.argv[1:]; root=pathlib.Path(args[args.index('--result-root')+1])\n"
            "(root/'args.json').write_text(json.dumps(args))\n"
            "(root/'summary.json').write_text(json.dumps({'counts':{'successful':1}}))\n"
        )
        info = {"cuda_graph_config": {"decode": {"backend": "full", "bs": [32]}},
                "dsv4_c4_cache_format": "native", "export_metrics_to_file": False}
        tools = repo / "tools"
        tools.mkdir()
        curl = tools / "curl"
        curl.write_text("#!/bin/sh\ncase \"$*\" in\n*server_info*) cat <<'JSON'\n" + json.dumps(info) + "\nJSON\n;;\n*flush_cache*) [ \"${FAIL_FLUSH:-0}\" != 1 ] || exit 22; echo 'Cache flushed.';;\nesac\n")
        curl.chmod(0o755)
        (local / "env.sh").write_text(
            f"HOST_REPO={shlex.quote(str(repo))}\nRESULTS_HOST=$HOST_REPO/results\n"
            "REPO_CT=/mnt/host_root$HOST_REPO\nLOG_HOST=$HOST_REPO/logs\nPORT=30381\nMODEL_NAME=fixture\n"
            "check_sources() { return 0; }\nts() { echo 20260101_000000; }\n"
        )
        (local / "eval-env.sh").write_text(f"REPLAY_RUNNER={shlex.quote(str(fake_runner))}\nREPLAY_ADAPTER=$REPLAY_RUNNER\nREPLAY_DATASET=$HOST_REPO\nREPLAY_MANIFEST=$REPLAY_RUNNER\nrequire_replay_config() {{ return 0; }}\n")
        environment = {"PATH": str(tools) + os.pathsep + os.environ["PATH"], "PYTHON": sys.executable}
        output = repo / "results/lswb-replay/explicit"
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                env = {**environment, **({"LSWB_RUN_ROOT": str(output), "REPLAY_MINIMUM_SUCCESS_RATE": ".5"} if explicit else {})}
                result = subprocess.run(["bash", str(local / "bench-lswb.sh"), "tag", "30381", "2", "1"], env=env,
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                root = output if explicit else repo / "results/lswb-replay/tag/20260101_000000"
                arguments = json.loads((root / "client/args.json").read_text())
                self.assertEqual(arguments[arguments.index("--minimum-success-rate") + 1], ".5" if explicit else "0.99")
        result = subprocess.run(["bash", str(local / "bench-lswb.sh"), "tag"],
                                env={**environment, "LSWB_RUN_ROOT": str(output)}, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be fresh", result.stderr)
        failed_root = repo / "results/lswb-replay/failed-flush"
        result = subprocess.run(["bash", str(local / "bench-lswb.sh"), "failed", "30381", "2", "1"],
                                env={**environment, "LSWB_RUN_ROOT": str(failed_root), "FAIL_FLUSH": "1"},
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prefix cache flush failed", result.stderr)
        self.assertFalse((failed_root / "client/args.json").exists())

    def _fake_sglang(self) -> dict[str, str]:
        tools = self.root / "fake-sglang"
        tools.mkdir(exist_ok=True)
        fake = tools / "sglang"
        fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
        fake.chmod(0o755)
        return {"PATH": str(tools) + os.pathsep + os.environ["PATH"], "DECODE_CFG": "{}"}

    def test_run_server_v41_modes_select_only_the_v41_layout(self):
        script = kernels.REPO_ROOT / "scripts/local/run-server.sh"
        environment = {
            **self._fake_sglang(), "MODEL_FAMILY": "dsv41",
            "REMNANT_INTEGRATION": "v0.5.22", "TP": "4", "EP": "4",
        }
        for mode in ("native", "direct"):
            result = subprocess.run(
                ["bash", str(script), mode, "/model", "127.0.0.1", "1"],
                env=environment, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            words = result.stdout.split()
            for flag, value in (
                ("--kv-cache-format", mode), ("--ep", "4"),
                ("--dsv4-attn-backend", "flashmla"),
                ("--reasoning-parser", "deepseek-v41"),
                ("--tool-call-parser", "deepseekv41"),
            ):
                self.assertEqual(words[words.index(flag) + 1], value)
            self.assertFalse([w for w in words if w.startswith("--dsv4-c4-")])

    def test_replay_direct_execution_skips_git_check_and_names_v41_modes(self):
        """LSWB_EXECUTION=direct needs no git checkout and reports the V4.1 layout."""
        repo = self.root / "direct"
        local = repo / "scripts/local"
        local.mkdir(parents=True)
        (local / "bench-lswb.sh").write_text((kernels.REPO_ROOT / "scripts/local/bench-lswb.sh").read_text())
        fake_runner = repo / "client.py"
        fake_runner.write_text(
            "import pathlib, sys\n"
            "args=sys.argv[1:]; root=pathlib.Path(args[args.index('--result-root')+1])\n"
            "(root/'summary.json').write_text('{}')\n"
        )
        tools = repo / "tools"
        tools.mkdir()
        curl = tools / "curl"
        curl.write_text("#!/bin/sh\ncase \"$*\" in\n*server_info*) cat \"$INFO\";;\n*flush_cache*) echo ok;;\nesac\n")
        curl.chmod(0o755)
        (local / "env.sh").write_text(
            f"HOST_REPO={shlex.quote(str(repo))}\nRESULTS_HOST=$HOST_REPO/results\n"
            "REPO_CT=/not/the/host\nLOG_HOST=$HOST_REPO/logs\nPORT=30381\nMODEL_NAME=fixture\n"
            "check_sources() { echo git checkout required >&2; return 1; }\nts() { echo 20260101_000000; }\n"
        )
        (local / "eval-env.sh").write_text(
            f"REPLAY_RUNNER={shlex.quote(str(fake_runner))}\nREPLAY_ADAPTER=$REPLAY_RUNNER\n"
            "REPLAY_DATASET=$HOST_REPO\nREPLAY_MANIFEST=$REPLAY_RUNNER\nrequire_replay_config() { return 0; }\n"
        )
        graphs = {"cuda_graph_config": {"decode": {"backend": "full", "bs": [48]}}, "export_metrics_to_file": False}
        environment = {"PATH": str(tools) + os.pathsep + os.environ["PATH"], "PYTHON": sys.executable}

        def run(info: dict, execution: str | None, run_root: str) -> subprocess.CompletedProcess:
            info_path = repo / "info.json"
            info_path.write_text(json.dumps(info))
            env = {**environment, "INFO": str(info_path), "LSWB_RUN_ROOT": str(repo / "results/lswb-replay" / run_root)}
            if execution:
                env["LSWB_EXECUTION"] = execution
            return subprocess.run(["bash", str(local / "bench-lswb.sh"), "tag", "30381", "2", "1"],
                                  env=env, capture_output=True, text=True, timeout=15)

        result = run({**graphs, "dsv4_c4_cache_format": "native"}, None, "docker-default")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("git checkout required", result.stderr)
        result = run({**graphs, "dsv4_c4_cache_format": "native"}, "bogus", "bogus")
        self.assertEqual(result.returncode, 2)
        self.assertIn("LSWB_EXECUTION must be docker or direct", result.stderr)
        for fmt in ("native", "direct"):
            with self.subTest(v41=fmt):
                info = {**graphs, "dsv4_c4_cache_format": "native", "kv_cache_format": fmt}
                result = run(info, "direct", f"direct-{fmt}")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                config = json.loads((repo / f"results/lswb-replay/direct-{fmt}/collection-config.json").read_text())
                self.assertEqual(config["mode"], fmt)
        result = run({**graphs, "dsv4_c4_cache_format": "native", "kv_cache_format": "auto"}, "direct", "direct-v4")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        config = json.loads((repo / "results/lswb-replay/direct-v4/collection-config.json").read_text())
        self.assertEqual(config["mode"], "native")


class LswbReplayPlanTests(unittest.TestCase):
    """CPU contracts for the Modal paired-LSWB launcher and its concurrency plan."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    @staticmethod
    def _manifest(sizes: list[list[int]]) -> dict:
        records = []
        for user, session in enumerate(sizes):
            for turn, tokens in enumerate(session):
                records.append({
                    "request_id": f"u{user:03d}-{turn:03d}", "user_id": f"user{user:03d}",
                    "created": float(user * 1000 + turn), "expected_prompt_tokens": tokens - 1,
                    "expected_completion_tokens": 1,
                })
        return {"records": records}

    def test_working_set_matches_a_hand_computed_schedule(self):
        from scripts.local.business_replay import capacity_plan as plan

        sizes = [[10, 20], [5, 5, 5], [100]]
        sessions = plan.session_sizes(self._manifest(sizes))
        self.assertEqual(sessions, sizes)
        # C=2: tick0 10+5, tick1 20+5, then session 0 ends and [100] joins: 100+5.
        self.assertEqual(plan.peak_working_set(sessions, 2, ticks=3), 105)
        self.assertEqual(plan.peak_working_set(sessions, 2, ticks=2), 25)
        self.assertEqual(plan.peak_working_set(sessions, 1, ticks=3), 20)
        with self.assertRaises(ValueError):
            plan.peak_working_set(sessions, 0)

    def test_plan_picks_largest_fitting_c_and_rejects_tiny_pools(self):
        from scripts.local.business_replay import capacity_plan as plan

        sessions = [[1000] * 5 for _ in range(30)]  # Each live session holds 1000 tokens.
        chosen = plan.plan_concurrency(sessions, pool_tokens=20000, max_running=48, ticks=5)
        self.assertEqual((chosen["c_high"], chosen["c_low"]), (18, 14))  # 18_000 <= 0.9 * 20_000
        self.assertEqual(plan.plan_concurrency(sessions, 10**9, 12, ticks=5)["c_high"], 12)
        with self.assertRaisesRegex(ValueError, "minimum 8"):
            plan.plan_concurrency(sessions, 5000, 48, ticks=5)

    def test_server_info_must_describe_the_requested_arm(self):
        from scripts.modal import serve

        good = {
            "tp_size": 4, "ep_size": 4, "kv_cache_format": "native", "random_seed": serve.LSWB_SEED,
            "max_running_requests": serve.LSWB_MAX_RUN, "mem_fraction_static": 0.92,
            "cuda_graph_config": {"decode": {"bs": [1, 48, 136]}},
        }
        serve._check_server_info(good, "native")
        for key, value in (("kv_cache_format", "direct"), ("tp_size", 8),
                           ("random_seed", 1), ("max_running_requests", 256), ("ep_size", 1)):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                serve._check_server_info({**good, key: value}, "native")
        with self.assertRaisesRegex(RuntimeError, "decode graphs"):
            serve._check_server_info({**good, "cuda_graph_config": {"decode": {"bs": [1, 8]}}}, "native")

    def test_replay_stall_detection(self):
        from scripts.modal import serve

        responses = self.root / "responses.jsonl"
        self.assertIsNone(serve._replay_stalled(responses, 299))
        self.assertIn("first 300 s", serve._replay_stalled(responses, 301))
        rows = [{"valid": index >= 3} for index in range(60)]  # 3/60 = 5% fails: allowed.
        responses.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assertIsNone(serve._replay_stalled(responses, 600))
        rows[3]["valid"] = False  # 4/60 > 5%.
        responses.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assertIn("4/60", serve._replay_stalled(responses, 600))

    def test_lswb_arguments_are_validated_before_any_allocation(self):
        from scripts.modal import serve

        with patch.object(serve, "_check_devices") as devices:
            with patch.object(serve, "MODEL_FAMILY", "dsv4"), self.assertRaisesRegex(ValueError, "REMNANT_MODEL"):
                serve.benchmark_lswb.local("native")
            with patch.object(serve, "MODEL_FAMILY", "dsv41"):
                with self.assertRaisesRegex(ValueError, "mode must be"):
                    serve.benchmark_lswb.local("invalid")
                for low, high in ((10, 0), (0, 10), (20, 10), (10, 49)):
                    with self.subTest(low=low, high=high), self.assertRaisesRegex(ValueError, "c_low"):
                        serve.benchmark_lswb.local("native", c_low=low, c_high=high)
            devices.assert_not_called()


    def _run_lswb(self, mode: str, pool: int = 5_000_000, **kwargs):
        from scripts.modal import serve
        from scripts.local.business_replay import capacity_plan

        lswb = self.root / "lswb"
        (lswb / "replay").mkdir(parents=True, exist_ok=True)
        (lswb / "replay/manifest.json").write_text(json.dumps(self._manifest([[100_000] * 6 for _ in range(60)])))
        (lswb / "prepare-receipt.json").write_text("{}")
        info = {
            "tp_size": 4, "ep_size": 4, "kv_cache_format": serve.LSWB_SERVER_MODES[mode],
            "random_seed": serve.LSWB_SEED, "max_running_requests": serve.LSWB_MAX_RUN,
            "mem_fraction_static": 0.92, "cuda_graph_config": {"decode": {"bs": [136]}},
            "max_total_num_tokens": pool,
        }
        points = []
        with (
            patch.object(serve, "MODEL_FAMILY", "dsv41"), patch.object(serve, "LSWB_ROOT", lswb),
            patch.object(serve, "RESULTS_ROOT", self.root / "results"),
            patch.object(serve, "REMOTE_REPO", kernels.REPO_ROOT),
            patch.object(serve, "_kernel_extensions_ready", return_value=True),
            patch.object(serve, "_check_devices", return_value=["H200"] * 4),
            patch.object(serve, "_running_serving_server", return_value=nullcontext()) as boot,
            patch.object(serve, "_http_json", return_value=info),
            patch.object(serve, "_sanity_check", return_value=["ok"]),
            patch.object(serve, "_run_replay_point", side_effect=lambda *a: points.append(a[2:])),
            patch.object(serve.results_volume, "commit"),
        ):
            run = Path(serve.benchmark_lswb.local(mode, **kwargs))
        return run, points, boot

    def test_lswb_boots_once_and_replays_matched_points_low_then_high(self):
        run, points, boot = self._run_lswb("native")
        plan = json.loads((run / "c-plan.json").read_text())
        # Each live session holds 100K..600K tokens; the plan comes from the 5M-token pool.
        self.assertEqual([p[0] for p in points], [f"native-c{plan['c_low']}", f"native-c{plan['c_high']}"])
        self.assertEqual([p[1] for p in points], [plan["c_low"], plan["c_high"]])
        self.assertLess(plan["c_low"], plan["c_high"])
        self.assertLessEqual(plan["peak_high"], 0.9 * 5_000_000)
        boot.assert_called_once()
        env = boot.call_args.args[0]
        self.assertEqual((env["SERVING_MODE"], env["LSWB_EXECUTION"], env["MAX_RUN"], env["MEM_FRAC"]),
                         ("native", "direct", "48", "0.92"))
        self.assertEqual(boot.call_args.args[3], serve_markers := boot.call_args.args[3])
        self.assertIn("CUDA out of memory", serve_markers)

    def test_second_arm_reuses_given_c_and_fails_if_it_cannot_fit(self):
        run, points, _ = self._run_lswb("direct", c_low=2, c_high=3)
        self.assertEqual([p[1] for p in points], [2, 3])
        self.assertTrue(json.loads((run / "c-plan.json").read_text())["given"])
        with self.assertRaisesRegex(RuntimeError, "above the 3000000-token pool"):
            self._run_lswb("direct", pool=3_000_000, c_low=30, c_high=48)









class BuildLockTests(unittest.TestCase):
    """Real lock files in a temporary repository; no Modal process is started."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="remnant-lock-")
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)

    def test_live_owner_blocks_second_acquire_and_dead_owner_is_stale(self):
        parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(parent.wait)
        self.addCleanup(parent.kill)
        (self.repo / build_lock.LOCK_NAME).write_text(json.dumps({"pid": parent.pid}))
        self.assertEqual(build_lock.lock_owner(self.repo), parent.pid)
        with self.assertRaisesRegex(RuntimeError, "already running"):
            build_lock.acquire(self.repo)
        parent.kill()
        parent.wait()
        self.assertIsNone(build_lock.lock_owner(self.repo))
        with patch("atexit.register"):
            build_lock.acquire(self.repo)
        self.assertEqual(build_lock.lock_owner(self.repo), os.getpid())
        build_lock.release(self.repo)
        self.assertFalse((self.repo / build_lock.LOCK_NAME).exists())

    def test_only_the_modal_cli_takes_the_lock(self):
        self.assertIsNone(build_lock.acquire_for_modal_cli(self.repo, "/usr/bin/pytest"))
        self.assertFalse((self.repo / build_lock.LOCK_NAME).exists())
        with patch("atexit.register"):
            build_lock.acquire_for_modal_cli(self.repo, "/opt/venv/bin/modal")
        self.assertEqual(json.loads((self.repo / build_lock.LOCK_NAME).read_text())["pid"], os.getpid())
        build_lock.release(self.repo)
        with patch("atexit.register"):
            self.assertIsNotNone(build_lock.acquire_for_modal_cli(self.repo, "/site/modal/__main__.py"))
        self.assertIsNone(build_lock.acquire_for_modal_cli(self.repo, "/site/other/__main__.py"))

    def test_stale_locks_are_ignored_and_replaced(self):
        lock = self.repo / build_lock.LOCK_NAME
        for content in ("", "{not json", json.dumps({"pid": 0}), json.dumps({"pid": -1}), json.dumps({"x": 1})):
            lock.write_text(content)
            self.assertIsNone(build_lock.lock_owner(self.repo), content)
            with patch("atexit.register"):
                build_lock.acquire(self.repo)
            self.assertEqual(build_lock.lock_owner(self.repo), os.getpid())
            build_lock.release(self.repo)

    def test_reused_pid_with_different_start_time_is_stale(self):
        lock = self.repo / build_lock.LOCK_NAME
        lock.write_text(json.dumps({"pid": os.getpid(), "start": "Mon Jan  1 00:00:00 1990"}))
        self.assertIsNone(build_lock.lock_owner(self.repo))
        lock.write_text(json.dumps({"pid": os.getpid(), "start": build_lock._start_time(os.getpid())}))
        self.assertEqual(build_lock.lock_owner(self.repo), os.getpid())

    def test_simultaneous_launchers_admit_exactly_one(self):
        code = (
            "import sys, time\nfrom pathlib import Path\nfrom scripts import build_lock\n"
            "time.sleep(max(0.0, float(sys.argv[2]) - time.time()))  # release all racers together\n"
            "try:\n    build_lock.acquire(Path(sys.argv[1]))\nexcept RuntimeError:\n    print('REFUSED')\n"
            "else:\n    print('HELD'); sys.stdout.flush(); time.sleep(6)\n"
        )
        start = str(time.time() + 3)
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", code, str(self.repo), start],
                cwd=kernels.REPO_ROOT, stdout=subprocess.PIPE, text=True,
            )
            for _ in range(6)
        ]
        results = [p.communicate(timeout=30)[0].strip() for p in procs]
        self.assertEqual(sorted(results), ["HELD"] + ["REFUSED"] * 5)
        self.assertFalse((self.repo / build_lock.LOCK_NAME).exists(), "owner exit must release")

    def test_settings_register_the_hook_for_edit_tools(self):
        settings = json.loads((kernels.REPO_ROOT / ".claude/settings.json").read_text())
        (entry,) = settings["hooks"]["PreToolUse"]
        self.assertTrue({"Edit", "Write"} <= set(entry["matcher"].split("|")))
        (hook,) = entry["hooks"]
        self.assertEqual(hook["type"], "command")
        self.assertIn(".claude/hooks/deny-edit-during-build.py", hook["command"])
        self.assertTrue((kernels.REPO_ROOT / ".claude/hooks/deny-edit-during-build.py").is_file())

    def test_hook_denies_image_inputs_only_while_locked(self):
        hook = kernels.REPO_ROOT / ".claude/hooks/deny-edit-during-build.py"
        env = {**os.environ, "CLAUDE_PROJECT_DIR": str(self.repo)}
        (self.repo / ".claude/hooks").mkdir(parents=True)
        (self.repo / "scripts").mkdir()
        (self.repo / "scripts/build_lock.py").write_text((kernels.REPO_ROOT / "scripts/build_lock.py").read_text())

        def run(name):
            call = json.dumps({"tool_input": {"file_path": str(self.repo / name)}})
            return subprocess.run(
                [sys.executable, str(hook)], input=call, env=env, text=True, capture_output=True, timeout=15
            )

        self.assertEqual(run("scripts/modal/kernels.py").returncode, 0)
        (self.repo / build_lock.LOCK_NAME).write_text(json.dumps({"pid": os.getpid()}))
        blocked = run("scripts/modal/kernels.py")
        self.assertEqual(blocked.returncode, 2)
        self.assertIn("build is running", blocked.stderr)
        self.assertEqual(run("third_party/sglang/x.py").returncode, 2)
        for image_input in ("scripts/local/serve.sh", "Dockerfile.v0.5.22", ".dockerignore", "scripts/sources.py", "scripts/model_profiles.py"):
            self.assertEqual(run(image_input).returncode, 2, image_input)
        self.assertEqual(run("docs/report.md").returncode, 0)


class ValidationTierTests(unittest.TestCase):
    """Pure path classification; the tier is guidance, so these pin the rules."""

    def test_parent_paths(self):
        expected = {
            "docs/report.md": "low",
            "docs/agents/modal.md": "low",
            "docs/figures/a.png": "low",
            "AGENTS.md": "low",
            ".github/pull_request_template.md": "low",
            "transfer/run-1/SHA256SUMS": "low",
            "transfer/run-1/summary.json": "low",
            "benchmarks/microbench/x/runs/r1/result.json": "low",
            "third_party/README.md": "low",
            "scripts/modal/kernels.py": "medium",
            "scripts/local/serve.sh": "medium",
            "benchmarks/microbench/x/run_matrix.py": "medium",
            ".claude/settings.json": "medium",
            ".claude/commands/x.md": "medium",
            ".github/workflows/ci.yml": "medium",
            "requirements.txt": "medium",
            "transfer/x/run.sh": "medium",
            "scripts/modal/runs/a.py": "medium",
            "something/unrecognized.bin": "medium",
            "Dockerfile.v0.5.22": "high",
            "Dockerfile": "high",
            ".dockerignore": "high",
            ".gitmodules": "high",
            "docs/NOTES.MD": "low",
            "third_party/sglang-v0.5.22": "high",
            "third_party/flashmla-v0.5.18": "high",
        }
        for path, tier in expected.items():
            self.assertEqual(validation_tier.classify_path(path), tier, path)

    def test_fork_paths(self):
        expected = {
            "csrc/sm90/decode/sparse/splitkv_mla.cuh": "high",
            "CMakeLists.txt": "high",
            "sgl-kernel/CMakeLists.txt": "high",
            "python/sglang/srt/remnant/pack.py": "high",
            "python/sglang/srt/server_args.py": "high",
            "test/registered/attention/unittests/dsv4/test_remnant_pool.py": "medium",
            "python/sglang/benchmark/remnant/calibrate.py": "medium",
            "benchmark/remnant/microbench.py": "medium",
            "README.md": "low",
            "docs/notes.md": "low",
        }
        for path, tier in expected.items():
            self.assertEqual(validation_tier.classify_path(path, "fork"), tier, path)

    def test_overall_tier_is_the_highest_and_unknown_never_low(self):
        empty = validation_tier.classify([])
        self.assertEqual(empty["tier"], "low")
        self.assertIn("No changes found", empty["note"])
        self.assertEqual(validation_tier.classify(["docs/a.md"])["tier"], "low")
        self.assertEqual(validation_tier.classify(["docs/a.md", "scripts/sources.py"])["tier"], "medium")
        mixed = validation_tier.classify(["docs/a.md", "scripts/sources.py", "third_party/sglang-v0.5.18"])
        self.assertEqual(mixed["tier"], "high")
        self.assertEqual(mixed["required_checks"], validation_tier.REQUIRED_CHECKS["high"])
        self.assertEqual(validation_tier.classify(["new/place/file.py"])["tier"], "medium")

    def test_changed_paths_sees_renames_deletes_uncommitted_untracked_and_unicode(self):
        with tempfile.TemporaryDirectory(prefix="remnant-tier-") as temporary:
            repo = Path(temporary)

            def git(*arguments):
                subprocess.run(["git", "-C", str(repo), *arguments], check=True, capture_output=True, timeout=30)

            git("init", "-q", "-b", "main")
            git("config", "user.email", "t@example.invalid")
            git("config", "user.name", "t")
            (repo / "scripts/modal").mkdir(parents=True)
            (repo / "docs").mkdir()
            (repo / "scripts/modal/x.py").write_text("print(1)\n" * 20)
            (repo / "scripts/modal/gone.py").write_text("print(2)\n" * 20)
            (repo / "base.txt").write_text("base\n")
            git("add", "-A")
            git("commit", "-q", "-m", "base")
            git("switch", "-q", "-c", "work")
            git("mv", "scripts/modal/x.py", "docs/x.md")  # an executable moved to a doc path
            git("rm", "-q", "scripts/modal/gone.py")
            git("commit", "-q", "-am", "work")
            (repo / "base.txt").write_text("edited, uncommitted\n")
            (repo / "scripts/modal").mkdir(parents=True, exist_ok=True)
            (repo / "scripts/modal/untracked.py").write_text("print(3)\n")
            (repo / "docs/caf\u00e9.md").write_text("x\n")
            previous = os.getcwd()
            os.chdir(repo)
            try:
                paths = validation_tier.changed_paths("main")
            finally:
                os.chdir(previous)
            self.assertEqual(
                paths,
                sorted([
                    "base.txt", "docs/caf\u00e9.md", "docs/x.md", "scripts/modal/gone.py",
                    "scripts/modal/untracked.py", "scripts/modal/x.py",
                ]),
            )
            result = validation_tier.classify(paths)
            self.assertEqual(result["tier"], "medium")
            self.assertEqual(result["paths"]["scripts/modal/x.py"], "medium")
            self.assertEqual(result["paths"]["docs/caf\u00e9.md"], "low")

    def test_changed_paths_is_root_relative_from_any_subdirectory(self):
        with tempfile.TemporaryDirectory(prefix="remnant-tier-sub-") as temporary:
            repo = Path(temporary).resolve()

            def git(*arguments):
                subprocess.run(["git", "-C", str(repo), *arguments], check=True, capture_output=True, timeout=30)

            git("init", "-q", "-b", "main")
            git("config", "user.email", "t@example.invalid")
            git("config", "user.name", "t")
            (repo / "docs").mkdir()
            (repo / "scripts").mkdir()
            (repo / "docs/a.md").write_text("a\n")
            git("add", "-A")
            git("commit", "-q", "-m", "base")
            (repo / ".gitmodules").write_text("[submodule]\n")  # untracked, outside both subdirectories
            (repo / "scripts/new.py").write_text("print(1)\n")
            (repo / "docs/a.md").write_text("edited\n")
            expected = [".gitmodules", "docs/a.md", "scripts/new.py"]
            previous = os.getcwd()
            try:
                for subdirectory in (".", "docs", "scripts"):
                    os.chdir(repo / subdirectory)
                    with self.subTest(cwd=subdirectory):
                        paths = validation_tier.changed_paths("main")
                        self.assertEqual(paths, expected)
                        self.assertEqual(validation_tier.classify(paths)["tier"], "high")
            finally:
                os.chdir(previous)

    def test_cli_reports_a_bad_base_without_a_traceback(self):
        result = subprocess.run(
            [sys.executable, str(kernels.REPO_ROOT / "scripts/validation_tier.py"), "--base", "no-such-ref"],
            cwd=kernels.REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("validation_tier:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

class GLMUpstreamDecodeTests(unittest.TestCase):
    """Upstream dispatch and source-bound evidence; no remote calls or CUDA."""

    def test_public_dispatch_rejects_incompatible_scope_before_scheduling(self):
        base = dict(run_id="upstream", keep_count=256, modes="", suite="glm-upstream-decode",
                    prefill_reader="split", qualification_runs="qualified", model_family="glm53",
                    phase="decode", max_seconds=180)
        with (patch.object(kernels, "MODEL_FAMILY", "glm53"),
              patch.object(kernels, "_UPSTREAM_DECODE", True),
              patch.object(kernels, "_preflight_glm_timing") as preflight,
              patch.object(kernels, "_benchmark_worker") as worker):
            for changed in ({"phase": "all"}, {"modes": "native,direct"},
                            {"packed_prefill_tile": 64}, {"allow_incomplete_qualification": True},
                            {"max_seconds": 1}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    kernels._dispatch_benchmark(**{**base, **changed})
            preflight.remote.assert_not_called()
            worker.remote.assert_not_called()
            preflight.remote.side_effect = RuntimeError("qualification failed")
            with self.assertRaisesRegex(RuntimeError, "qualification failed"):
                kernels._dispatch_benchmark(**base)
            worker.remote.assert_not_called()
            preflight.remote.side_effect = None
            kernels._dispatch_benchmark(**base)
            self.assertEqual(preflight.remote.call_args.kwargs, {"suite": "glm-upstream-decode"})
            self.assertEqual(worker.remote.call_args.args[3], "glm-upstream-decode")

    def test_upstream_cpu_delta_owns_its_driver_stub_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "cpu-driver-stubs").mkdir()
            manifest = build / "manifest.json"
            manifest.write_text("{}")

            def environment(work):
                (work / "cpu-driver-stubs").mkdir()
                return {"driver": str(work)}

            with (patch.object(kernels, "_cpu_preflight_environment", side_effect=environment) as prepare,
                  patch.object(kernels, "_run_command"), patch.object(kernels, "_run_pytest"),
                  patch.object(kernels, "MANIFEST", manifest), patch.object(kernels, "Path") as paths):
                paths.return_value.read_text.return_value = '{"baseline": "compiled"}'
                kernels._prepare_upstream_decode(build, 100)
            self.assertEqual(prepare.call_args.args[0], build / "upstream")
            self.assertTrue((build / "cpu-driver-stubs").is_dir())
            self.assertEqual(json.loads(manifest.read_text())["upstream_compile"], {"baseline": "compiled"})

    def test_reused_scope_allows_harness_changes_but_keeps_compiler_and_cache_identity(self):
        import hashlib
        import xml.etree.ElementTree as ET

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "microbench/fixture/validation"
            folder.mkdir(parents=True)
            variants = [dict(heads=h, inner=i, packed=p, cubin_sha256=f"cubin-{h}-{i}-{p}")
                        for h in (8, 16) for i in (1, 2) for p in (False, True)]
            manifest = dict(sources={"harness": "old"}, nvcc="compiler",
                            glm_compile=dict(triton=[], tilelang=variants, attention_scale=.0625,
                                             cache_files={"cached-direct": "same"},
                                             scope_symbol_audit={"trace": "bound"}),
                            sanitizer={"binary_sha256": "sanitizer"})
            (folder / "build-manifest.json").write_text(json.dumps(manifest))
            trace = ("glm_packed_bytes WarpSync CTA barrier\n"
                     "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n")
            (folder / "scope-audit.log").write_text(trace)
            result = dict(model_family="glm53", status="passed", runtime_hashes={"kernel": "same"},
                          sanitizer_scope="glm-merge",
                          sanitizer_results={t: {"status": "passed"} for t in kernels.SANITIZER_ORDER},
                          scope_audit=dict(status="passed", filter="regex=glm_packed_(partial|bytes)",
                                           log_sha256=hashlib.sha256(trace.encode()).hexdigest(),
                                           entrypoints={"trace": "bound"}))
            (folder / "result.json").write_text(json.dumps(result))
            for tool in kernels.SANITIZER_ORDER:
                (folder / f"{tool}.log").write_text(trace if tool == "racecheck" else "========= ERROR SUMMARY: 0 errors\n")
                suite = ET.Element("testsuite")
                for name in sorted(qualification.GLM_DECODE_REQUIRED | {
                    "test_prefill_partial_graph[64]", "test_prefill_partial_graph_h8"
                }):
                    node = ET.SubElement(suite, "testcase", name=name,
                                         classname="registered.attention.unittests.test_glm5_remnant")
                    if name.startswith("test_decode_partial_graph["):
                        h, i = map(int, name.split("[")[1][:-1].split("-"))
                        props = ET.SubElement(node, "properties")
                        for key, value in {
                            "partial_cubins": {str(p): f"cubin-{h}-{i}-{p}" for p in (False, True)},
                            "decode_geometry": dict(heads=h, inner=i, banks=2, queries=1, tails=[0,1,2,3],
                                                    nondefault_stream=True, launched_partitions=1,
                                                    binary_partitions=66 // i),
                        }.items():
                            ET.SubElement(props, "property", name=key, value=json.dumps(value))
                ET.ElementTree(suite).write(folder / f"{tool}.xml")
            current = copy.deepcopy(manifest)
            current["sources"] = {"harness": "new"}
            def qualify(candidate):
                context = qualification.QualificationContext(root, candidate, {"kernel": "same"}, kernels.SANITIZER_ORDER)
                return qualification.qualify_glm_timing("fixture", context=context, phase="upstream-decode")
            self.assertEqual(qualify(current)["status"], "passed")
            for change in ("compiler", "cache", "kernel"):
                changed = copy.deepcopy(current)
                if change == "compiler":
                    changed["nvcc"] = "different"
                elif change == "cache":
                    changed["glm_compile"]["cache_files"] = {"different": "cache"}
                else:
                    changed["glm_compile"]["tilelang"][0]["cubin_sha256"] = "different"
                with self.subTest(change=change), self.assertRaises(RuntimeError):
                    qualify(changed)


if __name__ == "__main__":
    unittest.main()
