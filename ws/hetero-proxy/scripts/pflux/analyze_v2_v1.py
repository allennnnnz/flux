# hetero-proxy pflux: analysis of v2_hostboard_v1.py output (v2_<tag>.json for tags shm, shmce).
# Applies the predictions P1-P7 and decision rules R1-R3 registered in the header of v2_hostboard_v1.py.
# Usage: python3 ws/hetero-proxy/scripts/pflux/analyze_v2_v1.py <results dir>   (writes <dir>/analysis.txt)
import json
import os
import sys

W = 8
H = 5120
MS = (256, 1024, 2048, 4096, 8192)


def main():
    d = sys.argv[1]
    runs = {t: json.load(open(os.path.join(d, f"v2_{t}.json"))) for t in ("shm", "shmce")
            if os.path.exists(os.path.join(d, f"v2_{t}.json"))}
    L = []
    p = L.append
    p(f"pflux V2 analysis ({d}); tags {list(runs)}; times = rank-max of per-rank medians (ms)")
    for t, r in runs.items():
        bad_ag = [k for c in r["checks"] for k, v in c.items() if k.startswith("ag_") and v is not True]
        rs_err = max(v for c in r["checks"] for k, v in c.items() if k.startswith("rs_"))
        p(f"[{t}] NCCL {r['nccl']} env {r['env']}")
        p(f"[{t}] clock modal {r['clock']['modal']} min {r['clock']['min']} samples {r['clock']['n_samples']}; "
          f"rounds dropped (max over items) {max(v['dropped'] for v in r['results'].values())}")
        p(f"[{t}] NVLink Data Tx+Rx delta KiB per GPU {r['nvlink_delta_KiB']}")
        p(f"[{t}] correctness: board AG not bit-identical to NCCL AG: {bad_ag or 'none'}; "
          f"board RS max |diff|/max|ref| = {rs_err:.2e}")

    def ms(tag, name, M):
        return runs[tag]["results"][f"{name}|{M}"]["ms_rankmax"]

    names = sorted({k.split("|")[0] for k in next(iter(runs.values()))["results"]})
    ag_var = [n for n in names if n.startswith(("ag_board/", "ag_sm/"))]
    rs_var = [n for n in names if n.startswith(("rs_board/", "rs_sm/"))]
    tags = list(runs)

    def cons(name, M):  # our items run in both tags: take the slower (conservative)
        return max(ms(t, name, M) for t in tags)

    p("")
    p("Self-consistency (our items in both tags; max |shm/shmce - 1| over sizes >= 1024):")
    if len(tags) == 2:
        for n in ag_var + rs_var:
            dev = max(abs(ms("shm", n, M) / ms("shmce", n, M) - 1) for M in MS if M >= 1024)
            p(f"  {n:28s} {dev * 100:5.1f}%")
    p("")
    hdr = f"{'M':>5} {'MiB':>5} | " + " ".join(f"{('ncclAG_' + t):>11}" for t in tags) + " | " + \
          " ".join(f"{v.replace('ag_board/', ''):>12}" for v in ag_var) + " | best  ratio  busbw(nccl->ours)"
    p("All-gather (output [M, 5120] bf16):")
    p(hdr)
    best = {}
    for M in MS:
        mib = M * H * 2 / 2 ** 20
        nccl = {t: ms(t, "ag_nccl", M) for t in tags}
        ours = {v: cons(v, M) for v in ag_var}
        bv = min(ours, key=ours.get)
        nbest = min(nccl.values())
        best[("ag", M)] = (nbest, ours[bv], bv)
        bw = lambda t_ms: M * H * 2 * (W - 1) / W / (t_ms / 1e3) / 1e9
        p(f"{M:5d} {mib:5.1f} | " + " ".join(f"{nccl[t]:11.3f}" for t in tags) + " | " +
          " ".join(f"{ours[v]:12.3f}" for v in ag_var) +
          f" | {bv.replace('ag_board/', '')} {nbest / ours[bv]:5.2f}x  {bw(nbest):5.1f}->{bw(ours[bv]):5.1f} GB/s")
    p("")
    p("Reduce-scatter (input [M, 5120] bf16):")
    for M in MS:
        nccl = {t: ms(t, "rs_nccl", M) for t in tags}
        ours = {v: cons(v, M) for v in rs_var}
        bv = min(ours, key=ours.get)
        nbest = min(nccl.values())
        best[("rs", M)] = (nbest, ours[bv], bv)
        p(f"{M:5d} | nccl " + " ".join(f"{t}={nccl[t]:.3f}" for t in tags) + " | " +
          " ".join(f"{v.replace('rs_board/', '')}={ours[v]:.3f}" for v in rs_var) + f" | ratio {nbest / ours[bv]:.2f}x")
    p("")
    p("GEMM (cuBLAS bf16, F.linear, per-rank TP8 shapes; rank-max, conservative over tags):")
    gem = {}
    for M in MS:
        gem[M] = {g: cons(f"gemm/{g}", M) for g in ("qkv", "o_proj", "fc1", "fc2")}
        p(f"{M:5d} | " + " ".join(f"{g}={v:.3f}" for g, v in gem[M].items()) + f" | sum {sum(gem[M].values()):.3f}")
    p("")
    p("R1: best board AG vs best NCCL AG >= 1.2x at 20 MiB (M=2048) and 40 MiB (M=4096)")
    r1 = {M: best[("ag", M)][0] / best[("ag", M)][1] for M in (2048, 4096)}
    p("    " + ", ".join(f"M={M}: {v:.2f}x" for M, v in r1.items()) + f" -> {'PASS' if all(v >= 1.2 for v in r1.values()) else 'FAIL'}")
    p("R2: per-layer estimate (perfect overlap upper bound) >= 1.3x at M=4096; also no-overlap version")
    for M in MS:
        agn, agb, _ = best[("ag", M)]
        rsn, rsb, _ = best[("rs", M)]
        rsb = min(rsb, rsn)
        g = gem[M]
        base = 2 * agn + 2 * rsn + sum(g.values())
        upper = max(agb, g["qkv"]) + max(agb, g["fc1"]) + max(rsb, g["o_proj"]) + max(rsb, g["fc2"])
        noov = 2 * agb + 2 * rsb + sum(g.values())
        comm_share = (2 * agn + 2 * rsn) / base
        tag = " <- R2" if M == 4096 else ""
        p(f"    M={M:5d}: baseline {base:7.3f} ms (comm {comm_share * 100:4.1f}%), ours no-overlap {noov:7.3f} "
          f"({base / noov:4.2f}x), ours upper {upper:7.3f} ({base / upper:4.2f}x){tag}")
    M = 4096
    agn, agb, _ = best[("ag", M)]
    rsn, rsb, _ = best[("rs", M)]
    rsb = min(rsb, rsn)
    g = gem[M]
    base = 2 * agn + 2 * rsn + sum(g.values())
    upper = max(agb, g["qkv"]) + max(agb, g["fc1"]) + max(rsb, g["o_proj"]) + max(rsb, g["fc2"])
    p(f"    -> {'PASS' if base / upper >= 1.3 else 'FAIL'} ({base / upper:.2f}x)")
    out = "\n".join(L)
    print(out)
    open(os.path.join(d, "analysis.txt"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
