"""One CPU reference job at a time: a reference tool takes the box's CPU model lock itself.

A tool started under a holder (the CPU queue runner sets B70_CPU_MODEL_LOCK_HELD=1; a wrapper script that flocked
the lock is an ancestor with it open) runs on. A second job started beside one refuses at once instead of sharing
the cores. Off Linux (the Mac) there is no lock.
"""
import fcntl
import os
import sys

_held = None


def _ancestor_holds(path: str) -> bool:
    pid = os.getpid()   # itself too: a wrapper that flocked the lock and then exec-ed the tool
    while pid > 1:
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    if os.readlink(f"/proc/{pid}/fd/{fd}") == path:
                        return True
                except OSError:   # an fd closed meanwhile (the listing's own)
                    continue
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except OSError:
            return False
    return False


def hold() -> None:
    global _held
    if not sys.platform.startswith("linux") or os.environ.get("B70_CPU_MODEL_LOCK_HELD"):
        return
    path = os.environ.get("B70_CPU_LOCK", "/tmp/b70-cpu-model.lock")
    if _ancestor_holds(path):
        return
    f = open(path, "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another CPU reference job holds {path}: one runs at a time - queue this one "
                 "(tools/cpu_queue.sh) instead of starting it beside")
    _held = f
