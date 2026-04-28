"""In-process job registry for scenario runs launched from the UI."""
from __future__ import annotations

import subprocess
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

Status = Literal["queued", "running", "done", "failed"]


@dataclass
class JobInfo:
    job_id: str
    scenario_path: str
    expected_run_prefix: str
    status: Status = "queued"
    pid: int | None = None
    started_at: str = ""
    finished_at: str = ""
    returncode: int | None = None
    _log: deque = field(default_factory=lambda: deque(maxlen=200))

    def log_tail(self) -> list[str]:
        return list(self._log)


_registry: dict[str, JobInfo] = {}
_lock = threading.Lock()


def launch_scenario(yaml_path: Path, scenario_name: str) -> JobInfo:
    job_id = str(uuid.uuid4())
    info = JobInfo(
        job_id=job_id,
        scenario_path=str(yaml_path),
        expected_run_prefix=f"{scenario_name}_",
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    with _lock:
        _registry[job_id] = info

    proc = subprocess.Popen(
        ["activation-lab", "run", str(yaml_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    info.pid = proc.pid
    info.status = "running"

    def _stream():
        assert proc.stdout is not None
        for line in proc.stdout:
            with _lock:
                info._log.append(line.rstrip())
        proc.wait()
        with _lock:
            info.returncode = proc.returncode
            info.status = "done" if proc.returncode == 0 else "failed"
            info.finished_at = datetime.now(timezone.utc).isoformat()

    threading.Thread(target=_stream, daemon=True).start()
    return info


def get(job_id: str) -> JobInfo | None:
    return _registry.get(job_id)
