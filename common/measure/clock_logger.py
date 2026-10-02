"""Background SM-clock sampler for discarding down-clocked measurement rounds.

Extracted from experiments/2026-09-23-phase0-a100-nvlink-pcie/scripts/
flux_ag_gemm_baseline_v2.py. No pynvml on this machine, so it shells out to
nvidia-smi (~30 ms per sample). Measurement rounds are usually shorter than the
sampling period, so `min_clock_between` falls back to the most recent sample at
or before the window start.

Usage:
    with ClockLogger(local_rank) as clk:
        for i in range(rounds):
            t0 = time.perf_counter()
            ...  # timed work
            windows.append((t0, time.perf_counter()))
        clocks = [clk.min_clock_between(*w) for w in windows]
    modal = statistics.mode(c for c in clocks if c)
    kept = [i for i, c in enumerate(clocks) if c is None or c >= 0.95 * modal]
"""

from __future__ import annotations

import subprocess
import threading
import time


class ClockLogger:
    def __init__(self, device: int, period_s: float = 0.02):
        self.device, self.period = device, period_s
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits",
                     "-i", str(self.device)],
                    capture_output=True, text=True, timeout=2,
                )
                self.samples.append((time.perf_counter(), int(out.stdout.strip().split()[0])))
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self) -> "ClockLogger":
        self._t.start()
        return self

    def __exit__(self, *a) -> None:
        self._stop.set()
        self._t.join(timeout=3)

    def min_clock_between(self, t0: float, t1: float) -> int | None:
        """Min sample inside [t0, t1]; else the most recent sample at or before t0."""
        v = [c for t, c in self.samples if t0 <= t <= t1]
        if v:
            return min(v)
        prev = [c for t, c in self.samples if t <= t0]
        return prev[-1] if prev else None
