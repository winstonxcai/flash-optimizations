"""CPU-only checks for the shell entrypoint; no SGLang or GPU is started."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "mustafar/scripts/local/bench_serving.sh"


class ShellTests(unittest.TestCase):
    def test_model_revision_and_results_volume_are_not_mode_names(self):
        from mustafar.scripts.modal import app

        revision = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
        self.assertEqual(app.MODEL_REVISION, revision)
        tree = ast.parse(Path(app.__file__).read_text())
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "results_volume"
                for t in node.targets
            )
        )
        self.assertEqual(
            ast.literal_eval(assignment.value.args[0]), "mustafar-stage2a-results"
        )
        download = Mock(return_value="/models/test")
        with (
            patch.dict(
                sys.modules,
                {"huggingface_hub": SimpleNamespace(snapshot_download=download)},
            ),
            patch.object(app, "model_volume") as volume,
        ):
            self.assertEqual(app.download_model.local(), "/models/test")
            self.assertEqual(download.call_args.kwargs["revision"], revision)
            volume.commit.assert_called_once()

    def test_recorded_native_pool_field_names_are_preserved(self):
        text = (ROOT / "mustafar/tests/fixtures/modal/native-pool.log").read_text()
        self.assertIn("c4=1072832", text)
        self.assertIn("c4_state=26816", text)

    def test_syntax_and_help(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        result = subprocess.run(
            ["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("native|packed|fused", result.stdout)
        self.assertFalse(SCRIPT.with_suffix(".py").exists())

    def test_bad_arguments_fail_before_startup(self):
        for args in (
            ["bogus"],
            ["native", "0"],
            ["packed", "32768", "2048", "17"],
            ["native", "32768", "2048", "auto"],
        ):
            with self.subTest(args=args):
                result = subprocess.run(
                    ["bash", str(SCRIPT), *args], capture_output=True, check=False
                )
                self.assertEqual(result.returncode, 2)

    def test_modal_invokes_shell_and_persists_on_failure(self):
        from mustafar.scripts.modal import app

        with (
            patch.object(app.subprocess, "run") as run,
            patch.object(app, "results_volume") as volume,
        ):
            app.bench_serving.local(mode="packed", input_tokens=65536, concurrency=4)
            command = run.call_args.args[0]
            self.assertEqual(command[-4:], ["packed", "65536", "2048", "4"])
            self.assertTrue(command[-5].endswith("bench_serving.sh"))
            self.assertEqual(
                command[:4], ["timeout", "--signal=TERM", "--kill-after=30s", "60m"]
            )
            self.assertNotIn(
                "MEASURED_WAVES",
                {
                    k: v
                    for k, v in run.call_args.kwargs["env"].items()
                    if k not in app.os.environ
                },
            )
            volume.commit.assert_called_once()
            volume.reset_mock()
            run.side_effect = subprocess.CalledProcessError(1, command)
            with self.assertRaises(subprocess.CalledProcessError):
                app.bench_serving.local()
            volume.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
