# hetero-proxy pflux: analysis of v2c_rspush_v1.py output (v2c_shm.json). Applies C1-C4 and the recomputed R2
# registered in the header of v2c_rspush_v1.py. Usage: python3 analyze_v2c_v1.py <results dir>  (writes analysis.txt)
import json
import os
import sys

W, H = 8, 5120
MS = (256, 1024, 2048, 4096, 8192)
V2_REF = {"ag_nccl": 4.886, "ag_board/own/ovl/0": 3.418, "rs_nccl": 4.993, "rs_board/ovl": 5.694}  # V2, M=4096


def main():
    d = sys.argv[1]
    r = json.load(open(os.path.join(d, "v2c_shm.json")))
    ms = lambda n, M: r["results"][f"{n}|{M}"]["ms_rankmax"]
    L = []
    p = L.append
    p(f"pflux V2c analysis ({d}); NCCL {r['nccl']}; wait flags {r['wait_flags']} (0 = no remote-write flush support)")
    p(f"NVLink Data Tx+Rx delta KiB per GPU {r['nvlink_delta_KiB']}")
    errs = {k: v for c in r["checks"] for k, v in c.items() if k.startswith("rs_")}
    bad_ag = [k for c in r["checks"] for k, v in c.items() if k.startswith("ag_") and v is not True]
    p(f"C1 correctness: max |diff|/max|ref| rs_push {max(v for k, v in errs.items() if 'push' in k):.2e}, "
      f"rs_board {max(v for k, v in errs.items() if 'board' in k):.2e}; board AG not bit-identical: {bad_ag or 'none'}")
    p("")
    p(f"{'M':>5} | {'ag_nccl':>8} {'ag_board':>8} {'ratio':>6} | {'rs_nccl':>8} {'rs_board':>8} {'push_T1':>8} {'push_T2':>8} "
      f"{'T2 ratio':>8} | busbw RS nccl -> push_T2")
    for M in MS:
        rsb = M * H * 2 * (W - 1) / W
        p(f"{M:5d} | {ms('ag_nccl', M):8.3f} {ms('ag_board/own/ovl/0', M):8.3f} "
          f"{ms('ag_nccl', M) / ms('ag_board/own/ovl/0', M):5.2f}x | {ms('rs_nccl', M):8.3f} {ms('rs_board/ovl', M):8.3f} "
          f"{ms('rs_push/T1', M):8.3f} {ms('rs_push/T2', M):8.3f} {ms('rs_nccl', M) / ms('rs_push/T2', M):7.2f}x | "
          f"{rsb / ms('rs_nccl', M) / 1e6:5.2f} -> {rsb / ms('rs_push/T2', M) / 1e6:5.2f} GB/s")
    p("")
    M = 4096
    c2 = ms("rs_push/T2", M)
    p(f"C2 rs_push at M=4096 3.9-4.6 ms, 1.08-1.28x: {c2:.3f} ms, {ms('rs_nccl', M) / c2:.3f}x -> "
      f"{'HOLDS' if 3.9 <= c2 <= 4.6 else 'WRONG'} (lower edge of the ratio range)")
    c3 = all(ms("rs_push/T2", M_) < ms("rs_push/T1", M_) for M_ in MS if M_ >= 2048)
    p(f"C3 T=2 fastest at M >= 2048: {'HOLDS' if c3 else 'WRONG'} (T1 is 1.36-1.42x slower: one thread cannot keep up)")
    dev = {n: ms(n, 4096) / v - 1 for n, v in V2_REF.items()}
    p("C4 reference items vs V2 at M=4096: " + ", ".join(f"{n} {x * 100:+.1f}%" for n, x in dev.items())
      + f" -> {'HOLDS' if all(abs(x) <= 0.05 for x in dev.values()) else 'WRONG'}")
    p("")
    p("R2 recomputed (RS_b = min(rs_push, rs_board, rs_nccl); perfect-overlap upper bound; also no overlap):")
    for M in MS:
        g = {k: ms(f"gemm/{k}", M) for k in ("qkv", "o_proj", "fc1", "fc2")}
        agn, agb = ms("ag_nccl", M), ms("ag_board/own/ovl/0", M)
        rsn = ms("rs_nccl", M)
        rsb = min(ms("rs_push/T2", M), ms("rs_push/T1", M), ms("rs_board/ovl", M), rsn)
        base = 2 * agn + 2 * rsn + sum(g.values())
        upper = max(agb, g["qkv"]) + max(agb, g["fc1"]) + max(rsb, g["o_proj"]) + max(rsb, g["fc2"])
        noov = 2 * agb + 2 * rsb + sum(g.values())
        p(f"  M={M:5d}: baseline {base:7.3f} ms (comm {(2 * agn + 2 * rsn) / base * 100:4.1f}%), no-overlap {noov:7.3f} "
          f"({base / noov:4.2f}x), upper {upper:7.3f} ({base / upper:4.2f}x)" + (" <- R2" if M == 4096 else ""))
        if M == 4096:
            verdict = base / upper
    p(f"  -> R2 (recomputed) {'PASS' if verdict >= 1.3 else 'FAIL'} ({verdict:.2f}x at M=4096); "
      "R2 as first registered (V2 items only) stays FAIL (1.29x)")
    out = "\n".join(L)
    print(out)
    open(os.path.join(d, "analysis.txt"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
