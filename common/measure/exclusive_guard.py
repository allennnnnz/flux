# WITHDRAWN 2026-10-03: owner by truncated user name and lifetime-average CPU% (blind to bursts of long-lived processes; ws/fusion-dispatch/reports/20261003_gpu1_inventory.md section 4). Replaced by common/measure/exclusive_guard_v2.py.
"""Exclusive-machine guard for measurements (CLAUDE.md 5.1: numbers are only valid if nothing
else shares the GPUs / CPUs while they are taken).

Wraps any command:
    python3 common/measure/exclusive_guard.py --log <file> [--allow_busy] -- <command> [args...]

1. Preflight (abort with exit 3 unless --allow_busy):
   - any process already on any GPU (nvidia-smi --query-compute-apps)
   - processes of OTHER users using > --cpu_pct CPU
   Logged-in users other than the current one are reported (warning only; `who`).
2. While the command runs, every --period seconds:
   - GPU compute processes that are NOT descendants of the wrapped command  -> "FOREIGN_GPU"
   - other users' processes above --cpu_pct CPU                              -> "FOREIGN_CPU"
3. On exit: a summary line with counts and the first/last foreign sample time. Exit code = the
   command's exit code; the summary says CLEAN or CONTAMINATED (exact time windows in the log, so a
   run can be repeated or rounds in those windows discarded).
Added 2026-10-02 by ws/fusion-dispatch after the user's instruction to ensure exclusive use.
"""
import argparse
import os
import pwd
import subprocess
import sys
import threading
import time

ME = pwd.getpwuid(os.getuid()).pw_name


def gpu_pids():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout
    res = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0].isdigit():
            res.append((int(parts[0]), parts[1] if len(parts) > 1 else "?"))
    return res


def proc_table():
    """pid -> (ppid, user, cpu%) from ps (cpu% is lifetime average; good enough to spot hogs)."""
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,user=,pcpu="], capture_output=True, text=True).stdout
    t = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 4:
            t[int(p[0])] = (int(p[1]), p[2], float(p[3]))
    return t


def descends(pid, root, table):
    seen = 0
    while pid in table and seen < 64:
        if pid == root:
            return True
        pid = table[pid][0]
        seen += 1
    return pid == root


def other_users_logged_in():
    out = subprocess.run(["who"], capture_output=True, text=True).stdout
    return sorted({l.split()[0] for l in out.splitlines() if l.strip() and l.split()[0] != ME})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--cpu_pct", type=float, default=20.0)
    ap.add_argument("--allow_busy", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    log = open(a.log, "a")

    def w(msg):
        log.write(f"{time.strftime('%F %T')} {msg}\n")
        log.flush()

    w(f"START cmd={' '.join(cmd)}")
    busy = gpu_pids()
    table = proc_table()
    hogs = [(pid, u, c) for pid, (pp, u, c) in table.items() if u != ME and u != "root" and c > a.cpu_pct]
    users = other_users_logged_in()
    w(f"PREFLIGHT gpu_procs={busy} other_user_cpu_hogs={hogs} other_users_logged_in={users}")
    if (busy or hogs) and not a.allow_busy:
        w("ABORT machine not exclusive")
        print(f"[exclusive_guard] ABORT: GPU processes {busy}, other-user CPU hogs {hogs}", file=sys.stderr)
        sys.exit(3)
    child = subprocess.Popen(cmd)
    stats = {"gpu": 0, "cpu": 0, "first": None, "last": None}
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                t = proc_table()
                foreign = [(pid, mem, t.get(pid, (0, "?", 0))[1]) for pid, mem in gpu_pids()
                           if not descends(pid, child.pid, t)]
                hog = [(pid, u, c) for pid, (pp, u, c) in t.items() if u not in (ME, "root") and c > a.cpu_pct]
                if foreign or hog:
                    now = time.strftime('%F %T')
                    stats["first"] = stats["first"] or now
                    stats["last"] = now
                    if foreign:
                        stats["gpu"] += 1
                        w(f"FOREIGN_GPU {foreign}")
                    if hog:
                        stats["cpu"] += 1
                        w(f"FOREIGN_CPU {hog}")
            except Exception as exc:  # noqa: BLE001
                w(f"MONITOR_ERROR {exc}")
            stop.wait(a.period)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    rc = child.wait()
    stop.set()
    th.join(timeout=5)
    verdict = "CLEAN" if stats["gpu"] == 0 and stats["cpu"] == 0 else "CONTAMINATED"
    w(f"END rc={rc} {verdict} foreign_gpu_samples={stats['gpu']} foreign_cpu_samples={stats['cpu']} "
      f"window={stats['first']}..{stats['last']} users_logged_in_at_end={other_users_logged_in()}")
    print(f"[exclusive_guard] {verdict} (foreign GPU samples {stats['gpu']}, CPU {stats['cpu']}) log={a.log}", file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
