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
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

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

    def test_only_three_remote_functions_and_no_model_volumes(self):
        self.assertEqual(
            set(kernels.app.registered_functions), {"benchmark_kernels", "profile_kernels", "validate_kernels"}
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
                    argv[:5],
                    [sys.executable, "-m", "pytest", "-q", str(kernels.SGLANG_ROOT / kernels.BENCHMARK_TESTS[0])],
                )
                self.assertNotIn("--worker", argv)

    def test_validation_budget_is_bounded_before_execution(self):
        function = inspect.unwrap(kernels.validate_kernels.get_raw_f())
        self.assertEqual(function.__defaults__, (False, 1100, ""))
        with patch.object(kernels, "_run_context") as run:
            for seconds in (0, 59, 2641, 2700):
                with self.assertRaisesRegex(ValueError, "max_seconds"):
                    function("fixture", max_seconds=seconds)
            run.assert_not_called()
        with patch.object(kernels, "VALIDATION_TIMEOUT_S", 120):
            with self.assertRaisesRegex(ValueError, "between 60 and 60"):
                function("fixture", max_seconds=61)

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
                "synccheck": {"status": "passed"},
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
            patch.object(kernels.subprocess, "run") as command,
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
            self.assertEqual(record["sanitizer_complete"], complete)
            self.assertEqual(
                record["sanitizer_results"],
                {t: {"status": "passed" if t in ran else "not_selected"} for t in order},
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
        with patch.object(kernels.subprocess, "run") as process:
            with self.assertRaises(TimeoutError):
                kernels._run_command(["command"], self.root / "output", 0)
            process.assert_not_called()
            for failure in (
                subprocess.CalledProcessError(1, ["command"]),
                subprocess.TimeoutExpired(["command"], 30),
                FileNotFoundError("command"),
            ):

                def fail(*args, failure=failure, **kwargs):
                    kwargs["stdout"].write("specific compiler failure")
                    raise failure

                process.side_effect = fail
                with self.assertRaisesRegex(RuntimeError, "specific compiler failure") as raised:
                    kernels._run_command(["command"], self.root / "output", kernels.time.monotonic() + 30)
                self.assertIn(str(self.root / "output"), str(raised.exception))
                self.assertIs(raised.exception.__cause__, failure)
                self.assertLessEqual(process.call_args.kwargs["timeout"], 30)

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
            patch.object(serve, "_compiled_source_digest", return_value="matching"),
            patch.dict(os.environ, {"REMNANT_BUILT_INTEGRATION": serve.INTEGRATION["version"]}),
        ):
            serve._verify_compiled_sources("matching")
            with self.assertRaisesRegex(RuntimeError, "Compiled source changed"):
                serve._verify_compiled_sources("different")
            wrong = "v0.5.18" if serve.INTEGRATION["version"] == "v0.5.22" else "v0.5.22"
            with patch.dict(os.environ, {"REMNANT_BUILT_INTEGRATION": wrong}):
                with self.assertRaisesRegex(RuntimeError, "different REMNANT_INTEGRATION"):
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

    def test_public_commands_import_without_launching_jobs(self):
        code = """
from unittest.mock import patch
from pathlib import Path
import modal
with patch.object(modal.Function, 'remote', side_effect=AssertionError('launch')), \\
     patch.object(modal.Function, 'spawn', side_effect=AssertionError('launch')), \\
     patch.object(modal.Image, 'from_dockerfile', wraps=modal.Image.from_dockerfile) as recipe:
    from scripts.modal import serve, kernels
    expected = 'Dockerfile.' + serve.INTEGRATION['version']
    assert Path(recipe.call_args.args[0]).name == expected
    assert recipe.call_args.kwargs['build_args']['REMNANT_INTEGRATION'] == serve.INTEGRATION['version']
    assert 'benchmark_serving' in serve.app.registered_functions
    assert {'benchmark_kernels', 'profile_kernels', 'validate_kernels'} <= set(kernels.app.registered_functions)
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
                        "dsv4_c4_cache_implementation", "missing_implementation_flag"
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
        for function in (kernels.benchmark_kernels, kernels.profile_kernels):
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
            self.assertEqual(kernels.benchmark_kernels.local("count192", keep_count=192), str(self.root))
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
            with patch.object(kernels.subprocess, "run") as process:
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
            kernels.benchmark_kernels.local("prefill", suite="prefill-read")
        argv = command.call_args.args[0]
        self.assertEqual(argv[argv.index("--suite") + 1], "prefill-read")
        self.assertNotIn("--modes", argv)
        self.assertNotIn("--keep-count", argv)
        validate.assert_called_once_with(payload, suite="prefill-read")

    def test_prefill_invalid_options_fail_before_context(self):
        for function in (kernels.benchmark_kernels, kernels.profile_kernels):
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
            kernels.benchmark_kernels.local(
                "fused-reader", suite="prefill-read", prefill_reader="fused"
            )
        argv = command.call_args.args[0]
        self.assertEqual(argv[argv.index("--prefill-reader") + 1], "fused")
        self.assertEqual(record["prefill_reader"], "fused")
        validate.assert_called_once_with(
            payload, suite="prefill-read", prefill_reader="fused"
        )
        for function in (kernels.benchmark_kernels, kernels.profile_kernels):
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
        return self.sweep.parse_args(["kernel", "--concurrencies", "2,3", "--tag", "screen",
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
        env = {**self.env, "HICACHE_RATIO": "2", "SWEEP_MODE": "kernel"}
        with self.assertRaisesRegex(ValueError, "HICACHE_RESET_RECEIPT"):
            self.sweep.validate_hicache_receipt(env, self.image, self.sources)
        evidence = self.root / "reset.json"
        evidence.write_text("GPU reset observation")
        receipt = {"image_id": self.image, "integration": "v0.5.18", "mode": "kernel", "hicache_ratio": 2.,
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
        self.assertEqual([c[-1] for c in serve], ["kernel", "stop"])
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
                    if boot_fails and str(arguments[-1]) == "kernel":
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
                self.sweep.check_server({**self.info, **change}, {**self.env, "SERVER_SEED": "20260919"}, "kernel", 3)

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



if __name__ == "__main__":
    unittest.main()
