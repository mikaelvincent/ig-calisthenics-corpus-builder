from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestRunCommandWritesLog(unittest.TestCase):
    def test_run_creates_run_log_on_config_error(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            missing_cfg = Path(td) / "missing_config.yaml"

            env = dict(os.environ)
            env["APIFY_TOKEN"] = "dummy"
            env["OPENAI_API_KEY"] = "dummy"

            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else str(repo_root)
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ig_corpus",
                    "run",
                    "--config",
                    str(missing_cfg),
                    "--out",
                    str(out_dir),
                ],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 2, msg=proc.stderr)

            log_path = out_dir / "run.log"
            self.assertTrue(log_path.exists())

            lines = [
                ln.strip()
                for ln in log_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            self.assertGreaterEqual(len(lines), 1)

            events: list[str] = []
            for ln in lines:
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                ev = obj.get("event")
                if isinstance(ev, str):
                    events.append(ev)

            self.assertIn("run_command_started", events)
            self.assertIn("run_command_failed", events)


if __name__ == "__main__":
    unittest.main()
