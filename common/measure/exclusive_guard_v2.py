"""Exclusive-machine guard v2 (CLAUDE.md 5.1.11: numbers are only valid if nothing else shares the GPUs / CPUs
while they are taken). Replaces exclusive_guard.py (v1), which had two blind spots found on 2026-10-03
(ws/fusion-dispatch/reports/20261003_gpu1_inventory.md section 4):
  1. owner by user NAME from `ps user=`; names longer than 8 characters are truncated ("allenzh+"), so the
     guard's own processes counted as another user's.  v2 compares numeric uids read from /proc.
  2. CPU% from `ps pcpu` = average over the process lifetime; a long-lived process of another user that
     suddenly becomes busy is invisible (151.7% in top, 0.2% in ps).  v2 samples /proc/<pid>/stat every
     period and uses the CPU time consumed during that period (instantaneous).
v2 also watches root processes (v1 ignored them): on css-host-158 / 159 no root process uses more than
~0.5% of a core when idle, so a root process above --root_cpu_pct (default one core) is reported.

Wraps any command (one guard per node for multi-node runs; a run is CLEAN only if every node is CLEAN):
    python3 common/measure/exclusive_guard_v2.py --log <file> [--allow_busy] -- <command> [args...]

1. Preflight (one 1-second CPU sample; abort with exit 3 unless --allow_busy):
   - any process already on any GPU (nvidia-smi --query-compute-apps)
   - another user's process above --cpu_pct of a core, or a root process above --root_cpu_pct
   Logged-in users other than the current one are reported (warning only).
2. While the command runs, every --period seconds:
   - GPU compute process that is NOT a descendant of the wrapped command          -> FOREIGN_GPU
   - another user's process above --cpu_pct (instantaneous)                        -> FOREIGN_CPU
   - root process above --root_cpu_pct (instantaneous)                             -> FOREIGN_CPU_ROOT
   - own process outside the wrapped command above --self_cpu_pct (warning only)   -> SELF_CPU
3. On exit: END line with CLEAN / CONTAMINATED, sample counts and the contaminated time window.
   All log times are UTC (css-host-159's local time zone is America/New_York, css-host-158's is UTC).
   Exit code = the command's exit code.
Added 2026-10-03 by ws/fusion-dispatch.
"""
import argparse
import os
import pwd
import subprocess
import sys
import threading
import time

MY_UID = os.getuid()
ME = pwd.getpwuid(MY_UID).pw_name
HZ = os.sysconf("SC_CLK_TCK")


def gpu_pids():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout
    res = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0].isdigit():
            res.append((int(parts[0]), parts[1] if len(parts) > 1 else "?"))
    return res


def snapshot():
    """pid -> (ppid, uid, cpu_ticks, starttime, comm), read from /proc."""
    t = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
            comm = s[s.index("(") + 1:s.rindex(")")]
            rest = s[s.rindex(")") + 2:].split()
            ppid, ticks, start = int(rest[1]), int(rest[11]) + int(rest[12]), int(rest[19])
            uid = os.stat(f"/proc/{d}").st_uid
            t[int(d)] = (ppid, uid, ticks, start, comm)
        except (OSError, ValueError, IndexError):
            continue
    return t


def cpu_pct(prev, cur, dt):
    """pid -> % of one core used between two snapshots (same pid and start time in both)."""
    res = {}
    for pid, (pp, uid, ticks, start, comm) in cur.items():
        p = prev.get(pid)
        if p and p[3] == start:
            res[pid] = (ticks - p[2]) / HZ / dt * 100.0
    return res


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
    users = set()
    for line in out.splitlines():
        if line.strip():
            try:
                if pwd.getpwnam(line.split()[0]).pw_uid != MY_UID:
                    users.add(line.split()[0])
            except KeyError:
                users.add(line.split()[0])
    return sorted(users)


def classify(cur, pct, a, root_pid):
    """Return (foreign_cpu, foreign_root, self_cpu) lists of (pid, user, comm, pct)."""
    fc, fr, sc = [], [], []
    for pid, c in pct.items():
        pp, uid, _, _, comm = cur[pid]
        if pid == os.getpid():
            continue
        if uid == MY_UID:
            if c > a.self_cpu_pct and not (root_pid and descends(pid, root_pid, cur)):
                sc.append((pid, ME, comm, round(c, 1)))
        elif uid == 0:
            if c > a.root_cpu_pct:
                fr.append((pid, "root", comm, round(c, 1)))
        elif c > a.cpu_pct:
            try:
                name = pwd.getpwuid(uid).pw_name
            except KeyError:
                name = str(uid)
            fc.append((pid, name, comm, round(c, 1)))
    return fc, fr, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--cpu_pct", type=float, default=20.0, help="another user's process, %% of one core")
    ap.add_argument("--root_cpu_pct", type=float, default=100.0, help="root process, %% of one core")
    ap.add_argument("--self_cpu_pct", type=float, default=50.0, help="own process outside the command (warning)")
    ap.add_argument("--allow_busy", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    log = open(a.log, "a")
    host = os.uname().nodename

    def w(msg):
        log.write(f"{time.strftime('%F %T', time.gmtime())}Z [{host}] {msg}\n")
        log.flush()

    w(f"START guard=v2 uid={MY_UID} cmd={' '.join(cmd)}")
    busy = gpu_pids()
    s0, t0 = snapshot(), time.monotonic()
    time.sleep(1.0)
    s1, t1 = snapshot(), time.monotonic()
    fc, fr, sc = classify(s1, cpu_pct(s0, s1, t1 - t0), a, None)
    users = other_users_logged_in()
    w(f"PREFLIGHT gpu_procs={busy} foreign_cpu={fc} foreign_root={fr} self_cpu={sc} other_users_logged_in={users}")
    if (busy or fc or fr) and not a.allow_busy:
        w("ABORT machine not exclusive")
        print(f"[exclusive_guard_v2 {host}] ABORT: GPU {busy}, other-user CPU {fc}, root CPU {fr}", file=sys.stderr)
        sys.exit(3)
    child = subprocess.Popen(cmd)
    stats = {"gpu": 0, "cpu": 0, "root": 0, "self": 0, "first": None, "last": None}
    stop = threading.Event()

    def monitor():
        prev, tp = snapshot(), time.monotonic()
        while not stop.wait(a.period):
            try:
                cur, tc = snapshot(), time.monotonic()
                pct = cpu_pct(prev, cur, tc - tp)
                prev, tp = cur, tc
                foreign = [(pid, mem, cur.get(pid, (0, -1, 0, 0, "?"))[4]) for pid, mem in gpu_pids()
                           if not descends(pid, child.pid, cur)]
                fc, fr, sc = classify(cur, pct, a, child.pid)
                if foreign or fc or fr:
                    now = time.strftime('%F %TZ', time.gmtime())
                    stats["first"] = stats["first"] or now
                    stats["last"] = now
                if foreign:
                    stats["gpu"] += 1
                    w(f"FOREIGN_GPU {foreign}")
                if fc:
                    stats["cpu"] += 1
                    w(f"FOREIGN_CPU {fc}")
                if fr:
                    stats["root"] += 1
                    w(f"FOREIGN_CPU_ROOT {fr}")
                if sc:
                    stats["self"] += 1
                    w(f"SELF_CPU (warning) {sc}")
            except Exception as exc:  # noqa: BLE001
                w(f"MONITOR_ERROR {exc}")

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    rc = child.wait()
    stop.set()
    th.join(timeout=5)
    verdict = "CLEAN" if stats["gpu"] == stats["cpu"] == stats["root"] == 0 else "CONTAMINATED"
    w(f"END rc={rc} {verdict} foreign_gpu_samples={stats['gpu']} foreign_cpu_samples={stats['cpu']} "
      f"foreign_root_samples={stats['root']} self_cpu_warnings={stats['self']} window={stats['first']}..{stats['last']} "
      f"users_logged_in_at_end={other_users_logged_in()}")
    print(f"[exclusive_guard_v2 {host}] {verdict} (foreign GPU {stats['gpu']}, CPU {stats['cpu']}, root {stats['root']}; "
          f"self warnings {stats['self']}) log={a.log}", file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
