"""Fused-kernel time models.

AG+GEMM (Flux AGKernel, sm80): event simulation of the actual tile schedule.
  * Tile -> thread-block assignment is CUTLASS stream-K (3rdparty/cutlass/include/cutlass/gemm/
    threadblock/threadblock_swizzle_streamk.h: get_blocks / get_sk_blocks / constructor), ported
    line by line below. SK tiles are tile indices [0, sk_tiles) and their blocks are launched first;
    DP blocks follow (gemm_universal_streamk.h:1014-1044; same code in
    src/ag_gemm/sm80_all_gather_gemm.hpp:860-892).
  * Flux rotates the tile row index so each rank starts at its own rows
    (src/ag_gemm/sm80_all_gather_gemm_threadblock_swizzle.hpp, tile_m_offset).
  * Before each tile a block spins until every shard covering the tile's rows has arrived
    (sm80_all_gather_gemm.hpp:923-939).
  * Shards are pulled one copy at a time from rank+1, rank+2, ... on one stream
    (src/coll/ths_op/all_gather_op.cc:553-576), so remote shard i arrives at i * copy time.
  No overlap fraction is fitted: whether the kernel hides communication falls out of the schedule.
GEMM+RS (Flux GemmRS): the epilogue writes each tile to its owner rank, so the scatter runs
  alongside the GEMM; exposed scatter = T_scat^2 / (T_scat + T_gemm) (fully hidden when the GEMM is
  long, fully exposed when it is short), plus a fixed cost alpha_rs (barrier + local reduction).
"""
import heapq
from functools import lru_cache

SMEM_PER_SM = 167936  # A100: 164 KiB shared memory per SM


def occupancy(tm, tn, tk, stages, elem_bytes=2):
    return max(1, SMEM_PER_SM // (stages * (tm * tk + tn * tk) * elem_bytes))


def _get_sk_blocks(sk_tiles, ipt, avail, max_occ, allow_partial):
    savings, best = -(10 ** 18), 0
    if sk_tiles == 0:
        return 0, savings
    sk_iters = sk_tiles * ipt
    dp_equiv_iters = ipt * ((sk_tiles + avail - 1) // avail)
    lo = min(avail, sk_tiles + 1) if allow_partial else avail
    hi = min(avail * max_occ, sk_iters // 2)  # kMinItersPerSkBlock = 2
    for t in range(lo, hi + 1):
        skw = (t + avail - 1) // avail
        eq = ((sk_iters + t - 1) // t) * skw
        peers = ((t + sk_tiles - 1) // sk_tiles) + 1
        ic = 0.02 * peers * eq
        if t % sk_tiles == 0:
            peers, ic = t // sk_tiles, 0.0
        s = dp_equiv_iters - eq - int(2.0 * skw + ic + 2.0 * peers)
        if s >= savings:
            savings, best = s, t
    return best, savings


def _get_blocks(tiles, ipt, avail, occ):
    full = tiles // avail
    fwt = full * avail
    part = tiles - fwt
    if part == 0:
        return tiles, 0
    if full < occ:
        b, s = _get_sk_blocks(part, ipt, avail, occ - full, True)
        return (fwt, b) if s >= 0 else (tiles, 0)
    if occ > 1 and full % occ == occ - 1:
        b, s = _get_sk_blocks(part, ipt, avail, 1, True)
        if s >= 0:
            return fwt, b
    b, s = _get_sk_blocks(part + avail, ipt, avail, occ - ((full - 1) % occ), False)
    return (fwt - avail, b) if s >= 0 else (tiles, 0)


@lru_cache(maxsize=4096)
def ag_schedule(M, n, K, W, rank, tile, stages, sk, raster, sms=108):
    """Per-block work lists [(tile_row, iters), ...] in launch order, for one rank."""
    tm, tn, tk = tile
    occ = occupancy(tm, tn, tk, stages)
    tiled_m, tiled_n = -(-M // tm), -(-n // tn)
    tiles, ipt = tiled_m * tiled_n, -(-K // tk)
    avail = 1 if sk == "DP" else sms  # StreamkDP -> avail_sms = 1 (gemm_v2_ag_kernel.hpp:244-246)
    dp_tiles, sk_blocks = _get_blocks(tiles, ipt, avail, occ) if avail > 1 else (tiles, 0)
    sk_tiles = tiles - dp_tiles
    sk_waves, regions, ipb, big = 0, 1, 0, 0
    if sk_blocks > 0:
        sk_waves = -(-sk_blocks // avail)
        sk_iters = sk_tiles * ipt
        sk_blocks = min(sk_blocks, sk_iters)
        ipb, big = sk_iters // sk_blocks, sk_iters % sk_blocks
        if sk_blocks > sk_tiles and sk_blocks % sk_tiles == 0:
            regions = sk_tiles
    dp_blocks, dfw = dp_tiles, 1
    cM, cN = 8, 4  # kCohortCtasM / N
    tcm, tcn = -(-tiled_m // cM), -(-tiled_n // cN)
    cohort = False
    if raster != "N":  # raster_order 1 -> CUTLASS get_tile_offset (may use cohorts)
        in_range = True
        if sk_tiles > 0:
            cti = (sk_tiles - 1) // (cM * cN)
            cgm = cti // tcn
            cgn = tcn - 1 if cgm > 0 else cti % tcn
            in_range = not ((cgm + 1) * cM >= tiled_m or (cgn + 1) * cN >= tiled_n)
        cb = tcm * tcn * cM * cN
        if in_range and dp_blocks >= sms * occ * 2 and dp_blocks / cb > 0.85:
            cohort, dp_blocks = True, cb
    if not cohort and sk_waves > 0:
        ex = (sk_waves + -(-dp_tiles // avail)) % occ
        if dfw + ex <= dp_tiles // avail:
            dfw += ex
            dp_blocks -= ex * avail
    off = -(-(rank * (M // W)) // tm)

    def row(idx):
        m, nn = divmod(idx, tiled_n)
        if raster != "N":
            if tiled_m < tiled_n:
                nn, m = divmod(idx, tiled_m)
            if cohort:
                ct, b = divmod(idx, cM * cN)
                cgm, cgn = divmod(ct, tcn)
                bm, bn = divmod(b, cN)
                m, nn = cgm * cM + bm, cgn * cN + bn
        return (m + off) % tiled_m if m < tiled_m else m

    blocks = []
    if sk_blocks:
        spr = sk_blocks // regions
        ipr = (sk_tiles * ipt) // regions
        bpr_big = big // regions
        for raw in range(sk_waves * avail):
            if raw >= regions * spr:
                continue  # padding block
            bir, reg = divmod(raw, regions)
            begin = reg * ipr + bir * ipb
            nit = ipb
            if bir < bpr_big:
                begin += bir
                nit += 1
            else:
                begin += bpr_big
            pieces, it = [], begin + nit
            while it > begin:  # SK blocks walk their range from the end (tile_idx--)
                t = (it - 1) // ipt
                lo = max(begin, t * ipt)
                pieces.append((row(t), it - lo))
                it = lo
            blocks.append(tuple(pieces))
    first_dp = 0 if cohort else sk_tiles
    for d in range(dp_blocks):
        tidx, allot = first_dp + d, dfw
        if d >= avail:
            allot, tidx = 1, tidx + (dfw - 1) * avail
        pieces = tuple((row(tidx + a * avail), ipt) for a in range(allot) if tidx + a * avail < tiles)
        if pieces:
            blocks.append(pieces)
    return tuple(blocks), sms * occ, tm


def simulate(blocks, slots, tm, M, W, arrival, t_iter, kappa=0.0, t_comm_end=0.0):
    """Greedy in-order dispatch of blocks onto `slots` resident-block slots. arrival[c] = ready time
    of rank c's shard (relative to kernel start). Work done before t_comm_end (copies still in
    flight) runs (1 + kappa) times slower (copy traffic competes for memory bandwidth).
    Returns the kernel span."""
    mpr = M // W
    free = [0.0] * slots
    end = 0.0
    slow = 1.0 + kappa
    for pieces in blocks:
        t = heapq.heappop(free)
        for m, nit in pieces:
            c0, c1 = (m * tm) // mpr, min((m * tm + tm - 1) // mpr, W - 1)
            ready = max(arrival[c0:c1 + 1])
            if ready > t:
                t = ready
            d = nit * t_iter
            if kappa and t < t_comm_end:
                room = (t_comm_end - t) / slow  # base work that fits before the copies finish
                t = t + d * slow if d <= room else t_comm_end + (d - room)
            else:
                t += d
        heapq.heappush(free, t)
        if t > end:
            end = t
    return end


@lru_cache(maxsize=4096)
def _span_no_wait(key):
    """Kernel span in units of one k-iteration with every shard present (calibrates t_iter so the
    no-wait span equals the standalone gemm_only time, which already contains wave quantization)."""
    blocks, slots, tm = ag_schedule(*key)
    return simulate(blocks, slots, tm, key[0], key[3], [0.0] * key[3], 1.0)


def fused_ag_time(M, n, K, W, cfg, t_flux_ag, t_gemm, prm, ranks=None):
    """Fused AG+GEMM time (rank max). t_flux_ag, t_gemm: standalone Flux AllGather and Flux
    gemm_only times for this (M, n, K). prm:
      t_k0, local_bw  kernel start after op start = t_k0 + local shard bytes / local_bw
                      (nsys: 31.7 us + bytes / 838 GB/s, results/{v8,e2}_nsys parse files)
      d_tail          standalone-AG time after the last shard lands that the fused op does not pay
      kappa           slowdown of GEMM work that runs while copies are in flight
    Remote shard i (pull order rank+1, rank+2, ...) lands at t_k + i/(W-1) of the copy window.
    Limitation: a block's work rate is fixed, so when several blocks share an SM (occupancy > 1) and
    some of them are only waiting, the busy ones are not sped up; overlap is then underestimated."""
    t_k = prm["t_k0"] + (M // W) * K * 2 / prm["local_bw"]
    last = max(0.0, t_flux_ag - prm["d_tail"] - t_k)
    out = 0.0
    for r in ranks or sorted({0, W // 2 - 1, W - 1}):
        key = (M, n, K, W, r, tuple(cfg["tile"]), cfg["stages"], cfg["sk"], cfg["raster"])
        blocks, slots, tm = ag_schedule(*key)
        arr = [0.0] * W
        for i in range(1, W):
            arr[(r + i) % W] = last * i / (W - 1)
        span = simulate(blocks, slots, tm, M, W, arr, t_gemm / _span_no_wait(key), prm.get("kappa", 0.0),
                        last)
        out = max(out, t_k + span)
    return out


def fused_rs_time(M, N, W, t_gemm, alpha_rs, beta_scat):
    """GemmRS time: GEMM + exposed epilogue scatter + fixed cost. beta_scat in bytes/ms."""
    t_scat = (W - 1) / W * M * N * 2 / beta_scat
    return alpha_rs + t_gemm + t_scat * t_scat / (t_scat + t_gemm)
