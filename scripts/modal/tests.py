"""Local CPU checks for the thin Modal launchers. Author: Winston Cai.

Run: python -m unittest scripts.modal.tests
No image builds, remote calls, weights or GPUs. Build/execution boundaries are mocked.
"""

import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

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
        cuda_home = self.root / "cuda"
        stub = cuda_home / "lib64/stubs/libcuda.so"
        stub.parent.mkdir(parents=True)
        stub.write_bytes(b"CPU-only driver stub")
        calls = []

        def command(argv, output, deadline, *, environment=None):
            calls.append(argv)
            if argv == ["ncu", "--list-sections"]:
                return " ".join(kernels.NCU_SECTIONS)
            if "--query-metrics-mode" in argv:
                return kernels.PREFILL_METRICS
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
            "sglang.kernels.jit.utils": SimpleNamespace(
                override_jit_cuda_arch=lambda *a: nullcontext()
            ),
            "sglang.kernels.jit.utils.compile.cache": SimpleNamespace(
                find_prebuilt=lambda **kw: library
            ),
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
        configure = next(c for c in calls if c[:2] == ["cmake", "-S"])
        self.assertIn("-DSGL_KERNEL_ENABLE_FLASHMLA_SM100=OFF", configure)
        self.assertIn("-DENABLE_BELOW_SM90=OFF", configure)
        self.assertEqual(calls[0][:2], [sys.executable, "-c"])
        self.assertEqual(tests.call_count, 3)
        self.assertTrue(
            all(
                "cpu-driver-stubs" in c.kwargs["environment"]["LD_LIBRARY_PATH"]
                for c in tests.call_args_list
            )
        )
        receipt = json.loads(manifest.read_text())
        self.assertEqual(receipt["arch"], "sm_90a")
        self.assertEqual(len(receipt["binaries"]), 3)

    def test_cpu_and_sanitizer_selection_use_names_not_list_positions(self):
        self.assertEqual(
            kernels.CPU_TESTS,
            tuple(
                kernels.TEST_MODULES[n]
                for n in ("cache_format", "pack", "backend", "compressor_store")
            ),
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
        self.assertEqual(set(kernels.SANITIZER_TESTS), expected)
        self.assertEqual(len(kernels.SANITIZER_TESTS), len(expected))
        # Check actual node identities without importing CUDA-dependent modules.
        for selector in kernels.SANITIZER_TESTS:
            path, *names = selector.split("::")
            node = ast.parse((kernels.SGLANG_SOURCE / path).read_text())
            for name in names:
                matches = [
                    child for child in node.body if getattr(child, "name", None) == name
                ]
                self.assertEqual(len(matches), 1, selector)
                node = matches[0]

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


class LocalScriptTests(unittest.TestCase):
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
                patch.object(kernels, "_run_command"),
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
            patch.object(kernels, "_run_command") as command,
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


if __name__ == "__main__":
    unittest.main()
