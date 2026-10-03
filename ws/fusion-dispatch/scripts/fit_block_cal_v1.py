################################################################################
# fusion-dispatch G4: fit the layout-model profile (predictor/block.py BlockProfile) from
# calibrate_block_v1.py output: vLLM all-reduce curve + RMSNorm / residual-add costs, per mode.
# Usage: python3 fit_block_cal_v1.py <calibration dir>   -> hw_profiles/css-host-158_tp<W>_<mode>_block.json
################################################################################
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "common", "cost_model"))
from predictor.block import BlockProfile  # noqa: E402

d = sys.argv[1]
meta = json.load(open(os.path.join(d, "meta_block_cal.json")))
W = meta["world"]
rows = list(csv.DictReader(open(os.path.join(d, "summary_block_cal.csv"))))
for mode in ("gpu", "steady"):
    R = [r for r in rows if r["mode"] == mode]
    ar = [(int(r["M"]) * int(r["H"]) * 2, float(r["median_ms"])) for r in R if r["item"] == "vllm_ar"]
    nm = [(int(r["M"]), int(r["H"]), float(r["median_ms"])) for r in R if r["item"] == "rmsnorm"]
    ad = [(int(r["M"]), int(r["H"]), float(r["median_ms"])) for r in R if r["item"] == "add"]
    bp = BlockProfile.fit(W, ar, nm, ad, meta={"source": os.path.relpath(d, REPO), "date": time.strftime("%Y-%m-%d"),
                                              "method": "calibrate_block_v1.py (vLLM venv), H=6144, M=8..16384",
                                              "custom_ar": meta.get("custom_ar")})
    err = [abs(bp.rmsnorm(m, h) / t - 1) for m, h, t in nm] + [abs(bp.add(m, h) / t - 1) for m, h, t in ad]
    out = os.path.join(REPO, "common", "cost_model", "hw_profiles", f"css-host-158_tp{W}_{mode}_block.json")
    bp.save(out)
    print(f"[tp{W} {mode}] wrote {os.path.relpath(out, REPO)}  norm t0={bp.norm_p[0] * 1e3:.1f}us bw={bp.norm_p[1] * 1e3 / 1e9:.0f}GB/s  "
          f"add t0={bp.add_p[0] * 1e3:.1f}us bw={bp.add_p[1] * 1e3 / 1e9:.0f}GB/s  elementwise fit MAPE {sum(err) / len(err) * 100:.1f}%  "
          f"AR 8MiB={bp.ar(8 << 20):.4f}ms")
