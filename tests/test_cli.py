"""Tests for the command-line interface."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

from cachesim.cli import main
from cachesim.config import DEFAULT_CONFIG
from cachesim.workloads import sequential, write_trace


class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.trace = os.path.join(self._tmp.name, "seq.trace")
        write_trace(self.trace, sequential(buffer_bytes=8 * 1024, passes=2))

    def run_main(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as exc:
                if isinstance(exc.code, int):
                    code = exc.code
                else:
                    err.write(str(exc.code))
                    code = 1
        return code, out.getvalue(), err.getvalue()

    def test_run_prints_a_report(self) -> None:
        code, out, _ = self.run_main(["run", self.trace])
        self.assertEqual(code, 0)
        self.assertIn("CACHE HIERARCHY SIMULATION REPORT", out)
        self.assertIn("Total accesses :        2,048", out)
        self.assertIn("AMAT (analytic formula)", out)

    def test_run_json(self) -> None:
        code, out, _ = self.run_main(["run", "--format", "json", self.trace])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["accesses"], 2048)
        self.assertEqual([lv["name"] for lv in data["levels"]], ["L1", "L2", "L3"])

    def test_run_with_config_file(self) -> None:
        path = os.path.join(self._tmp.name, "one.json")
        cfg = {
            "memory_access_time": 50,
            "levels": [DEFAULT_CONFIG["levels"][0]],
        }
        with open(path, "w") as f:
            json.dump(cfg, f)
        code, out, _ = self.run_main(["run", "--config", path, "--format", "json", self.trace])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(len(data["levels"]), 1)
        self.assertEqual(data["memory_access_time"], 50)

    def test_invalid_config_is_reported_without_traceback(self) -> None:
        path = os.path.join(self._tmp.name, "bad.json")
        with open(path, "w") as f:
            f.write('{"memory_access_time": 1, "levels": [{"name": "L1"}]}')
        code, _, err = self.run_main(["run", "--config", path, self.trace])
        self.assertEqual(code, 1)
        self.assertIn("levels[0] (L1): missing required key(s)", err)

    def test_malformed_json_config(self) -> None:
        path = os.path.join(self._tmp.name, "bad.json")
        with open(path, "w") as f:
            f.write("{not json")
        code, _, err = self.run_main(["run", "--config", path, self.trace])
        self.assertEqual(code, 1)
        self.assertIn("cannot load config", err)

    def test_missing_trace(self) -> None:
        code, _, err = self.run_main(["run", os.path.join(self._tmp.name, "nope.trace")])
        self.assertEqual(code, 1)
        self.assertIn("No such file", err)

    def test_empty_trace(self) -> None:
        path = os.path.join(self._tmp.name, "empty.trace")
        with open(path, "w") as f:
            f.write("# nothing\n")
        code, _, err = self.run_main(["run", path])
        self.assertEqual(code, 1)
        self.assertIn("no accesses", err)

    def test_gen_traces_writes_selected_files(self) -> None:
        out_dir = os.path.join(self._tmp.name, "gen")
        code, out, _ = self.run_main(["gen-traces", "--out-dir", out_dir, "conflict"])
        self.assertEqual(code, 0)
        self.assertEqual(os.listdir(out_dir), ["conflict.trace"])
        self.assertIn("65,536 accesses", out)

    def test_gen_traces_reports_an_unwritable_out_dir(self) -> None:
        blocker = os.path.join(self._tmp.name, "blocker")
        with open(blocker, "w") as handle:
            handle.write("not a directory\n")
        code, _, err = self.run_main(
            ["gen-traces", "--out-dir", os.path.join(blocker, "gen"), "conflict"]
        )
        self.assertEqual(code, 1)
        self.assertIn("error: cannot write", err)
        self.assertNotIn("Traceback", err)

    def test_python_dash_m_propagates_the_exit_code(self) -> None:
        """``python -m cachesim`` must exit like the console script does."""
        env = {**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(__file__))}
        ok = subprocess.run(
            [sys.executable, "-m", "cachesim", "--version"], capture_output=True, env=env
        )
        self.assertEqual(ok.returncode, 0)
        self.assertTrue(ok.stdout.startswith(b"cachesim "))
        bad = subprocess.run(
            [sys.executable, "-m", "cachesim", "run", os.path.join(self._tmp.name, "nope")],
            capture_output=True,
            env=env,
        )
        self.assertEqual(bad.returncode, 1)
        self.assertIn(b"error:", bad.stderr)

    def test_version(self) -> None:
        code, out, _ = self.run_main(["--version"])
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("cachesim "))


if __name__ == "__main__":
    unittest.main(verbosity=2)
