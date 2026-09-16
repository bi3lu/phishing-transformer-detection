"""Process lifetime, CV recovery and fail-closed orchestration regressions."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main as pipeline


class IsolationTests(unittest.TestCase):
    def test_identity_diagnostic_reports_exact_nested_difference(self) -> None:
        saved = {"environment": {"device": "mps", "platform": "macOS-26.6.2", "packages": {"torch": "2.10.0"}}}
        current = {"environment": {"device": "mps", "platform": "macOS-27.0", "packages": {"torch": "2.10.0"}}}
        self.assertEqual(
            pipeline.identity_differences(saved, current),
            ["environment.platform: saved='macOS-26.6.2'; current='macOS-27.0'"],
        )
        self.assertEqual(pipeline.identity_differences(saved, saved), [])

    def test_environment_change_is_rejected_without_rewriting_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "run_manifest.json"
            payload = {
                "protocol": "group_refit_v2",
                "seed": 42,
                "implementation": "same",
                "environment": {"device": "mps", "platform": "macOS-26.6.2"},
            }
            original = json.dumps(payload)
            manifest.write_text(original)
            (root / "completed.json").write_text("{}")
            with (
                patch.object(pipeline, "RESULTS_DIR", root),
                patch.object(pipeline, "RANDOM_STATE", 42),
                patch("src.utils.artifacts.implementation_identity", return_value="same"),
                patch(
                    "src.utils.artifacts.environment_metadata",
                    return_value={"device": "mps", "platform": "macOS-27.0"},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "environment.platform"):
                    pipeline.check_run_identity()
            self.assertEqual(manifest.read_text(), original)

    def test_fresh_run_diagnostic_does_not_create_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new-run"
            with patch.object(pipeline, "RESULTS_DIR", root):
                pipeline.check_run_identity()
            self.assertFalse(root.exists())

    def test_real_workers_have_separate_pids_and_inherit_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").write_text(
                "import os,sys\n"
                "fd=int(sys.argv[sys.argv.index('--worker-lock-fd')+1])\n"
                "os.fstat(fd)\n"
                "with open('pids.txt','a') as out: out.write(str(os.getpid())+'\\n')\n"
            )
            with (
                (root / "lock").open("w") as lock,
                patch.object(pipeline, "BASE_DIR", root),
                patch.object(pipeline, "RESULTS_DIR", root / "results"),
            ):
                pipeline.run_isolated("finetune", "first", lock.fileno())
                pipeline.run_isolated("finetune", "second", lock.fileno())
            pids = [int(value) for value in (root / "pids.txt").read_text().splitlines()]
            self.assertEqual(len(set(pids)), 2)
            self.assertNotIn(os.getpid(), pids)
            self.assertEqual(len(list((root / "results" / "execution").glob("*.json"))), 2)

    def test_cv_retries_oom_only_after_new_completed_fold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv = root / "cross_validation" / "model" / "template_group"
            cv.mkdir(parents=True)

            def finish_fold_then_fail(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                done = cv / "fold_1" / "completed.json"
                done.parent.mkdir()
                done.write_text('{"preserved": true}')
                (cv / "status.json").write_text(json.dumps({"status": "failed", "error": "MPS backend out of memory"}))
                return subprocess.CompletedProcess([], 1)

            with (
                patch.object(pipeline, "RESULTS_DIR", root),
                patch("main.subprocess.run", side_effect=finish_fold_then_fail) as run,
            ):
                # The next attempt has no new progress: do not retry a second time.
                def attempt(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                    if run.call_count == 1:
                        return finish_fold_then_fail()
                    return subprocess.CompletedProcess([], 1)

                run.side_effect = attempt
                with self.assertRaises(subprocess.CalledProcessError):
                    pipeline.run_isolated("kfold", "model", 0)
                self.assertEqual(run.call_count, 2)
                self.assertEqual((cv / "fold_1" / "completed.json").read_text(), '{"preserved": true}')

    def test_non_oom_failure_is_not_retried_even_with_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv = root / "cross_validation" / "model" / "source_holdout"
            cv.mkdir(parents=True)

            def fail(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                done = cv / "fold_1" / "completed.json"
                done.parent.mkdir()
                done.write_text("{}")
                (cv / "status.json").write_text(json.dumps({"status": "failed", "error": "Corrupt data"}))
                return subprocess.CompletedProcess([], 1)

            with (
                patch.object(pipeline, "RESULTS_DIR", root),
                patch("main.subprocess.run", side_effect=fail) as run,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    pipeline.run_isolated("source-holdout", "model", 0)
                self.assertEqual(run.call_count, 1)

    def test_cv_coordinator_schedules_baseline_once_then_each_model(self) -> None:
        with patch.object(pipeline, "run_isolated") as run:
            pipeline.execute_steps(["kfold", "source-holdout"], ["a", "b"], 7, worker=False)
        self.assertEqual(
            [call.args for call in run.call_args_list],
            [(step, model, 7) for step in ("kfold", "source-holdout") for model in ("baseline", "a", "b")],
        )


if __name__ == "__main__":
    unittest.main()
