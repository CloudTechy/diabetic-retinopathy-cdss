"""
A pytest run that uses the API test client must exit when it finishes.

The module-level aiosqlite engine in conftest.py keeps its in-memory
connection pooled, and aiosqlite runs each connection on a non-daemon worker
thread. Undisposed, that thread outlives the session and the interpreter
waits on it forever: the run prints its summary and never returns, which a CI
job reports as a timeout rather than a pass.
"""

import os
import subprocess
import sys
import textwrap

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_a_session_using_the_test_client_exits(tmp_path):
    probe = tmp_path / "test_probe_client.py"
    probe.write_text(textwrap.dedent("""
        def test_health(test_client):
            assert test_client.get("/health").status_code in (200, 503)
    """), encoding="utf-8")
    # A separate run so this session's own teardown is what gets observed; the
    # probe's conftest re-exports the suite's fixtures and session hooks.
    (tmp_path / "conftest.py").write_text(
        "from tests.conftest import *  # noqa: F401,F403\n", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=BACKEND + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(probe), "-q", "-p", "no:cacheprovider",
         "--rootdir", str(tmp_path)],
        cwd=BACKEND, env=env, capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stdout + proc.stderr
