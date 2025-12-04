from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestCLISmoke(unittest.TestCase):
    def test_dry_run_offline_cli(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.yaml"
            cfg_path.write_text("{}", encoding="utf-8")

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
                    "dry-run",
                    "--config",
                    str(cfg_path),
                    "--offline",
                ],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("scraped_count=", proc.stdout)
            self.assertIn("processed_count=", proc.stdout)
            self.assertIn("eligible_count=", proc.stdout)
            self.assertIn("example_decision=", proc.stdout)


if __name__ == "__main__":
    unittest.main()
