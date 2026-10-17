"""Local CPU checks for the thin Modal launchers. Author: Winston Cai.

Run: python -m unittest scripts.modal.tests
No image builds, remote calls, weights or GPUs. Build/execution boundaries are mocked.
"""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.modal import kernels


class ContractTests(unittest.TestCase):
    def test_profile_matrix_is_complete(self):
        matrix = kernels._benchmark_cases().selected_profile_matrix()
        self.assertEqual(len(matrix), 30)
        self.assertEqual(len(set(matrix)), 30)
        self.assertEqual(sum(suite == "decode" for suite, _, _ in matrix), 24)
        self.assertEqual({mode for _, _, mode in matrix}, {"native", "triton", "kernel"})

    def test_representative_profile_matrix(self):
        cases = "h64_b16_k512,h64_b16_k317,decode-r24-all_valid,prefill-r8192-all_valid"
        matrix = kernels._benchmark_cases().selected_profile_matrix(cases)
        self.assertEqual(len(matrix), 12)
        self.assertEqual({name for _, name, _ in matrix}, set(cases.split(",")))
        for name in cases.split(","):
            self.assertEqual(
                {mode for _, case, mode in matrix if case == name},
                {"native", "triton", "kernel"},
            )
        for invalid in ("unknown", "h64_b16_k512,h64_b16_k512"):
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

    def test_collection_and_sanitizer_use_existing_pytest(self):
        with patch.object(kernels, "_run_command") as command:
            kernels._run_pytest(kernels.VALIDATION_TESTS, self.root, "collect", 100, collect=True)
            self.assertIn("--collect-only", command.call_args.args[0])
            (self.root / "memcheck.xml").write_text('<testsuite tests="1"/>')
            prefix = ("compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "1")
            kernels._run_pytest(kernels.SANITIZER_TESTS, self.root, "memcheck", 100, prefix=prefix)
            self.assertEqual(command.call_args.args[0][:5], list(prefix))

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
        sglang, flashmla = self.root / "sglang", self.root / "flashmla"
        aot = sglang / "python/sglang/kernels/aot"
        package = self.root / "site/sgl_kernel"
        cache = self.root / "jit"
        library = cache / "module/key/valid/native.so"
        for directory in (aot / "python/sgl_kernel", flashmla, package, library.parent):
            directory.mkdir(parents=True)
        (aot / "python/sgl_kernel/flash_mla.py").write_text("# fork binding")
        native, manifest = self.root / "native.so", self.root / "manifest.json"
        calls = []

        def command(argv, output, deadline):
            calls.append(argv)
            if argv == ["ncu", "--list-sections"]:
                return " ".join(kernels.NCU_SECTIONS)
            if argv[:2] == ["cmake", "--build"]:
                for target in ("remnant_ops", "flashmla_ops"):
                    binary = self.root / "build/cmake" / (target + ".so")
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_bytes(target.encode())
            if "--check-imports" in argv:
                self.assertTrue(native.is_file())
                self.assertEqual((package / "flash_mla.py").read_text(), "# fork binding")
            return "test toolchain"

        def native_factory(*args):
            self.assertEqual(args, ("fp32", 512, 64, 64, False))
            library.write_bytes(b"native")

        modules = {
            "torch": SimpleNamespace(
                float32="fp32",
                __version__="test",
                version=SimpleNamespace(cuda="13.0"),
                utils=SimpleNamespace(cmake_prefix_path="/torch"),
            ),
            "sglang.kernels.jit.utils": SimpleNamespace(override_jit_cuda_arch=lambda *a: nullcontext()),
            "sglang.kernels.jit.utils.compile.cache": SimpleNamespace(find_prebuilt=lambda **kw: library),
            "sglang.kernels.ops.attention.dsv4.compress": SimpleNamespace(
                _jit_compress_norm_rope_module=native_factory
            ),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(kernels, "SGLANG_ROOT", sglang),
            patch.object(kernels, "FLASHMLA_ROOT", flashmla),
            patch.object(kernels, "MANIFEST", manifest),
            patch.object(kernels, "NATIVE_LIBRARY", native),
            patch.object(kernels, "BUILD_ROOT", self.root / "build"),
            patch.object(kernels.sysconfig, "get_paths", return_value={"purelib": str(package.parent)}),
            patch.dict(kernels.os.environ, {"SGLANG_JIT_CACHE_DIR": str(cache)}),
            patch.object(kernels, "_run_command", side_effect=command),
            patch.object(kernels, "_run_pytest") as tests,
        ):
            kernels._build_image()
        self.assertIn("-DSGL_KERNEL_ENABLE_FLASHMLA_SM100=OFF", calls[0])
        self.assertIn("-DENABLE_BELOW_SM90=OFF", calls[0])
        self.assertEqual(tests.call_count, 3)
        receipt = json.loads(manifest.read_text())
        self.assertEqual(receipt["arch"], "sm_90a")
        self.assertEqual(len(receipt["binaries"]), 3)

    def test_cpu_and_sanitizer_selection_use_names_not_list_positions(self):
        self.assertEqual(
            kernels.CPU_TESTS,
            tuple(kernels.TEST_MODULES[n] for n in ("cache_format", "pack", "backend", "compressor_store")),
        )
        self.assertTrue(kernels.SANITIZER_TESTS[0].startswith(kernels.TEST_MODULES["unpack_kernel"] + "::"))
        self.assertTrue(kernels.SANITIZER_TESTS[1].startswith(kernels.TEST_MODULES["cuda_graph"] + "::"))

    def test_cached_serving_image_rejects_mismatched_sources(self):
        from scripts.modal import serve

        with patch.object(serve, "_compiled_source_digest", return_value="matching"):
            serve._verify_compiled_sources("matching")
            with self.assertRaisesRegex(RuntimeError, "Compiled source changed"):
                serve._verify_compiled_sources("different")
        source = self.root / "source"
        source.mkdir()
        kernel = source / "kernel.cu"
        kernel.write_text("original")
        before = serve._compiled_source_digest([source])
        kernel.write_text("changed")
        self.assertNotEqual(before, serve._compiled_source_digest([source]))

    def test_serving_missing_extensions_never_compiles_on_gpu(self):
        from scripts.modal import serve

        with (
            patch.object(serve, "_kernel_extensions_ready", return_value=False),
            patch.object(serve.subprocess, "run") as process,
        ):
            with self.assertRaisesRegex(RuntimeError, "GPU compilation is disabled"):
                serve.benchmark_serving.local()
            process.assert_not_called()

    def test_model_free_jobs_mount_no_model_volume(self):
        for function in kernels.app.registered_functions.values():
            self.assertNotIn("/models", function.spec.volumes)

    def test_public_commands_import_without_launching_jobs(self):
        code = """
from unittest.mock import patch
import modal
with patch.object(modal.Function, 'remote', side_effect=AssertionError('launch')), \\
     patch.object(modal.Function, 'spawn', side_effect=AssertionError('launch')):
    from scripts.modal import serve, kernels
    assert 'benchmark_serving' in serve.app.registered_functions
    assert {'benchmark_kernels', 'profile_kernels', 'validate_kernels'} <= set(kernels.app.registered_functions)
"""
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=kernels.REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
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


if __name__ == "__main__":
    unittest.main()
