################################################################################
# hetero-proxy KV tier K2 (2026-10-16): multi-user multi-turn serving run on the K2 engine (k2_engine_v1.py).
# Question: with many users chatting and a GPU that holds only some conversations' KV, how much does keeping the
# evicted KV in host memory (CPU "device") help the whole service, and does the PCIe traffic slow the others?
# Users (k2_trace_v1.py: real ShareGPT conversations, replies of fixed length R_k): N seats; seat i serves
# conversations i, i + N, i + 2N, ... of the trace (identical for every policy). A user sends a turn, the engine
# answers with exactly R_k tokens, the user thinks Exp(mean --think) seconds (seeded per seat / conversation / turn),
# sends the next turn; after its last turn the seat starts its next conversation (also after a think time).
# Engine slots: --slots conversations' KV fit on the GPU (scaled down from the ~250 x 4k this machine holds - state
# this when reporting). Scheduler (rank 0 decides, plans broadcast over gloo, every rank executes the same plan):
#   each iteration: at most one prefill (FCFS waiting queue) then one decode step of every running sequence
#   (the newly prefilled one included); the device is synchronised at the end of every iteration
#   a turn needs a slot: its own (still resident -> "hit": extend from the cached KV), else a free one, else the
#   least recently used slot of a THINKING user (evicted); if none, the turn waits
#   miss handling = the policy:  recompute  prefill the whole history from position 0
#                                tier       restore the saved KV layer by layer, overlapped with the new turn (K1)
#                                ideal      slots = seats: nothing is ever evicted (upper bound)
#   tier saves every finished turn's new KV to the seat's host region right away (write-through, copy stream)
# Records per turn (arrival, kind, P, T, TTFT) and per iteration (prefill / decode GPU ms, batch, bytes moved over
# PCIe in that iteration, wall duration); a turn counts when it arrives after --warmup seconds.
# Modes: check (engine correctness on the HF multi-turn reference: new / ext / restore / batched decode) and run.
# Usage (8 ranks, repo root, inside exclusive_guard_v2 / tmux):
#   env -u CUDA_DEVICE_MAX_CONNECTIONS CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml \
#     ~/.venvs/hetero-proxy-hf/bin/python -m torch.distributed.run --nproc_per_node 8 \
#     ws/hetero-proxy/scripts/kvtier/k2_serve_v1.py --mode run --policy tier --out <dir> [--seats 96 --slots 24 ...]
################################################################################
import argparse
import json
import math
import os
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "e2e"))
from qwen_tp_v1 import init  # noqa: E402
from qwen_tp_v5 import pin  # noqa: E402
from k2_engine_v1 import K2Engine, HostKV, T_BUCKETS, bucket  # noqa: E402

MAX_LEN = 4608   # slot length: conversations <= 4096 tokens + padding of the last turn's bucket
MAX_TOTAL = 4096


# ---------------------------------------------------------------------------------------------- execution (all ranks)
def execute(m, host, plan):
    """Run one iteration's plan; returns (prefill GPU ms or None, decode GPU ms or None)."""
    cs = torch.cuda.current_stream()
    for slot, r, a, e in plan["saves"]:
        host.save(slot, r, a, e)
    e0, e1, e2, e3 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
    e0.record()
    pre = plan["pre"]
    if pre is not None:
        kind, slot, P, ids = pre["kind"], pre["slot"], pre["P"], pre["ids"]
        if host is not None:
            host.before_write(slot)
        if kind in ("new", "recompute"):
            m.run_new(ids, slot)
        elif kind == "hit":
            m.run_ext(ids, P, slot, gates_ready=True)
        else:  # restore
            host.restore_begin()
            m.run_ext(ids, P, slot, gates_ready=False)
            host.restore_issue(pre["region"], slot, 0, P)
    e1.record()
    e2.record()
    if plan["dec"]:
        m.run_decode(plan["dec"])
    e3.record()
    torch.cuda.synchronize()
    return (e0.elapsed_time(e1) if pre is not None else None), (e2.elapsed_time(e3) if plan["dec"] else None)


# ---------------------------------------------------------------------------------------------- scheduler (rank 0)
class Seat:
    def __init__(self, i, convs, args):
        self.i, self.convs, self.args = i, convs, args
        self.c = -1
        self.slot = None
        self.next_conversation(0.0)

    def think(self):
        return random.Random(self.args.seed * 10 ** 9 + self.i * 10 ** 5 + self.c * 100 + self.turn).expovariate(
            1.0 / self.args.think)

    def next_conversation(self, now):
        self.c += 1
        conv = self.convs[self.c % len(self.convs)]
        self.turns, self.turn = conv["turns"], 0
        self.ids = list(self.turns[0]["ids"])
        self.kv_len = self.saved_len = 0
        self.state = "think"
        self.t_next = now + (random.Random(self.args.seed * 7 + self.i).uniform(0, self.args.think) if self.c == 0
                             else self.think())


class Scheduler:
    def __init__(self, users, args, n_slots):
        self.args = args
        S = args.seats
        self.seats = [Seat(i, users[i::S], args) for i in range(S)]
        self.slots = [None] * n_slots
        self.last_used = [0.0] * n_slots
        self.wait, self.running, self.saves = [], [], []
        self.turn_log, self.iter_log = [], []

    def plan(self, now):
        for s in self.seats:
            if s.state == "think" and s.t_next <= now:
                s.state, s.t_arrive = "wait", s.t_next
                self.wait.append(s)
        pl = dict(saves=self.saves, pre=None, dec=[], stop=False)
        self.saves = []
        if self.wait and len(self.running) < self.args.max_b:
            # first waiting user that can run now: its KV is still resident, or a slot is free / evictable
            # (strict FCFS could deadlock: the head waits for a slot held by resident users queued behind it)
            free = [k for k, o in enumerate(self.slots) if o is None]
            idle = [k for k, o in enumerate(self.slots) if o is not None and o.state == "think"]
            u = next((w for w in self.wait if w.slot is not None or free or idle), None)
            if u is not None:
                slot = u.slot
                if slot is None:
                    if free:
                        slot = free[0]
                    else:
                        slot = min(idle, key=lambda k: self.last_used[k])
                        v = self.slots[slot]
                        v.slot = None
                        if self.args.policy != "tier":
                            v.kv_len = 0
                self.wait.remove(u)
                if u.slot is not None:
                    kind, P = "hit", u.kv_len
                elif u.turn == 0:
                    kind, P = "new", 0
                elif self.args.policy == "tier":
                    kind, P = "restore", u.saved_len
                else:
                    kind, P = "recompute", 0
                ids = u.ids[P:]
                u.slot, self.slots[slot] = slot, u
                pl["pre"] = dict(kind=kind, slot=slot, P=P, ids=ids, region=u.i)
                u.kind, u.P, u.T, u.t_start = kind, P, len(ids), now
                u.first_pos = P + len(ids)
                u.pos_next, u.gen, u.R = u.first_pos, 1, u.turns[u.turn]["R"]
                u.state = "run"
                self.running.append(u)
        for r in self.running:
            pl["dec"].append((r.slot, r.pos_next))
            r.pos_next += 1
            r.gen += 1
        return pl

    def finish(self, m, now):
        done = [r for r in self.running if r.gen >= r.R]
        self.running = [r for r in self.running if r.gen < r.R]
        for r in done:
            toks = m.tok_log[r.slot, r.first_pos:r.first_pos + r.R].tolist()
            r.ids += toks
            r.kv_len = r.pos_next  # = first_pos + R - 1: the KV of every token but the last reply token
            self.last_used[r.slot] = now
            self.turn_log[r.log_i]["t_done"] = now
            if self.args.policy == "tier":
                self.saves.append((r.slot, r.i, r.saved_len, r.kv_len))
                r.saved_len = r.kv_len
            r.turn += 1
            if r.turn < len(r.turns):
                r.ids += r.turns[r.turn]["ids"]
                r.state, r.t_next = "think", now + r.think()
            else:
                self.slots[r.slot], r.slot = None, None
                r.next_conversation(now)


def serve(m, host, rank, args, users, gloo):
    sch = Scheduler(users, args, m.n_slots) if rank == 0 else None
    t0 = time.time()
    end = args.warmup + args.duration
    pcie = {}
    while True:
        if rank == 0:
            now = time.time() - t0
            if now >= end:
                plan = dict(stop=True)
            else:
                plan = sch.plan(now)
                if plan["pre"] is None and not plan["dec"] and not plan["saves"]:
                    nxt = min((s.t_next for s in sch.seats if s.state == "think"), default=now + 0.002)
                    time.sleep(max(0.0, min(0.002, nxt - now)))
                    continue
            obj = [plan]
        else:
            obj = [None]
        dist.broadcast_object_list(obj, src=0, group=gloo)
        plan = obj[0]
        if plan["stop"]:
            break
        it_start = time.time() - t0 if rank == 0 else 0.0
        pre_ms, dec_ms = execute(m, host, plan)
        if rank == 0:
            now = time.time() - t0
            pre = plan["pre"]
            nb = sum((e - a) * 32 * 1024 for _, _, a, e in plan["saves"])
            rb = pre["P"] * 32 * 1024 if pre is not None and pre["kind"] == "restore" else 0
            sch.iter_log.append(dict(t0=it_start, t1=now, pre_ms=pre_ms, dec_ms=dec_ms, B=len(plan["dec"]),
                                     kind=pre["kind"] if pre else None, save_bytes=nb, restore_bytes=rb))
            if pre is not None:
                u = sch.slots[pre["slot"]]
                u.log_i = len(sch.turn_log)
                sch.turn_log.append(dict(seat=u.i, conv=u.c, turn=u.turn, kind=pre["kind"], P=pre["P"], T=len(pre["ids"]),
                                         R=u.R, t_arrive=u.t_arrive, t_start=it_start, ttft_ms=1e3 * (it_start - u.t_arrive) + pre_ms,
                                         pre_ms=pre_ms))
            sch.finish(m, now)
    if rank == 0:
        with open(os.path.join(args.out, f"serve_{args.policy}_{args.tag}.json"), "w") as f:
            json.dump(dict(args=vars(args), turns=sch.turn_log, iters=sch.iter_log, numa=host.numa if host else None),
                      f)
        tl = [t for t in sch.turn_log if t["t_arrive"] >= args.warmup and "t_done" in t]
        kinds = {}
        for t in tl:
            kinds.setdefault(t["kind"], []).append(t["ttft_ms"])
        il = [i for i in sch.iter_log if i["t0"] >= args.warmup]
        gen = sum(i["B"] for i in il) + sum(1 for i in il if i["kind"])
        print(f"[serve] {args.policy}: {len(tl)} turns after warm-up, generated {gen} tokens in {args.duration:.0f} s "
              f"= {gen / args.duration:.0f} tok/s", flush=True)
        allt = sorted(t["ttft_ms"] for t in tl)
        if allt:
            print(f"[serve] TTFT all: median {statistics.median(allt):.1f} ms, p90 {allt[int(0.9 * len(allt))]:.1f}, "
                  f"p99 {allt[min(len(allt) - 1, int(0.99 * len(allt)))]:.1f}", flush=True)
        for k, v in sorted(kinds.items()):
            v = sorted(v)
            print(f"[serve]   {k:<9} n={len(v):<5} median {statistics.median(v):8.1f} ms  p90 {v[int(0.9 * len(v))]:8.1f}",
                  flush=True)
        dec = [i["dec_ms"] for i in il if i["dec_ms"] is not None]
        if dec:
            print(f"[serve] decode step GPU ms median {statistics.median(dec):.2f}; iteration wall median "
                  f"{1e3 * statistics.median([i['t1'] - i['t0'] for i in il]):.2f} ms", flush=True)


# ---------------------------------------------------------------------------------------------- check (all ranks)
def check(m, host, rank, args):
    ref = json.load(open(args.reference))
    items = [it for it in ref["items"] if len(it["prompt_ids"]) + len(it["hf_tokens"]) <= MAX_TOTAL]
    rows = []

    def decode_alone(slot, pos0, n):
        for s in range(n - 1):
            m.run_decode([(slot, pos0 + s)])
            torch.cuda.synchronize()

    def tokens(slot, pos0, n):
        return m.tok_log[slot, pos0:pos0 + n].tolist()

    def noise(slot):
        m.kc[:, slot].normal_()
        m.vc[:, slot].normal_()
        torch.cuda.synchronize()

    for j, it in enumerate(items):
        ids, P, hf = it["prompt_ids"], it["prefix_len"], it["hf_tokens"]
        n, L0 = len(hf), len(ids)
        s = 3 + j
        res = {}
        noise(s)
        m.run_new(ids, s)
        decode_alone(s, L0, n)
        res["new"] = tokens(s, L0, n)
        noise(s)
        m.run_new(ids[:P], s)
        m.run_ext(ids[P:], P, s, gates_ready=True)
        decode_alone(s, L0, n)
        res["ext"] = tokens(s, L0, n)
        noise(s)
        m.run_new(ids[:P], s)
        host.save(s, 0, 0, P)
        torch.cuda.synchronize()
        noise(s)
        host.before_write(s)
        host.restore_begin()
        m.run_ext(ids[P:], P, s, gates_ready=False)
        host.restore_issue(0, s, 0, P)
        decode_alone(s, L0, n)
        res["restore"] = tokens(s, L0, n)
        out = {k: dict(tokens=v, vs_hf=next((i for i, (a, b) in enumerate(zip(v, hf)) if a != b), n)) for k, v in
               res.items()}
        for k in out:
            i = out[k]["vs_hf"]
            out[k]["hf_gap_at_divergence"] = it["hf_gaps"][i] if i < n else None
        out["restore"]["identical_to_ext"] = res["restore"] == res["ext"]
        rows.append(dict(name=it["name"], P=P, L=L0, slot=s, results=out))
        if rank == 0:
            print(f"[check] {it['name']} (P {P}, L {L0}, slot {s}): " + "  ".join(
                f"{k}: HF {v['vs_hf']}/{n} (gap {v['hf_gap_at_divergence']})" for k, v in out.items())
                  + f"  restore==ext {out['restore']['identical_to_ext']}", flush=True)
    # batched decode: every item at once (one slot each) vs alone
    for j, it in enumerate(items):
        noise(3 + j)
        m.run_new(it["prompt_ids"], 3 + j)
    torch.cuda.synchronize()
    n = min(len(it["hf_tokens"]) for it in items)
    for st in range(n - 1):
        m.run_decode([(3 + j, len(it["prompt_ids"]) + st) for j, it in enumerate(items)])
        torch.cuda.synchronize()
    for j, it in enumerate(items):
        bt = tokens(3 + j, len(it["prompt_ids"]), n)
        same = next((i for i, (a, b) in enumerate(zip(bt, rows[j]["results"]["new"]["tokens"])) if a != b), n)
        rows[j]["batched_vs_alone_identical_prefix"] = same
        if rank == 0:
            print(f"[check] batched decode ({len(items)} rows) vs alone, {it['name']}: identical first {same}/{n}",
                  flush=True)
    if rank == 0:
        with open(os.path.join(args.out, "check_k2_v1.json"), "w") as f:
            json.dump(rows, f, indent=1)


def main():
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ.get("HXB_STACK_DUMP_S", "0")) or 10 ** 7, repeat=True, exit=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="tier", choices=["recompute", "tier", "ideal"])
    ap.add_argument("--users", default="ws/hetero-proxy/results/kvtier/k2/users_v1.json")
    ap.add_argument("--reference", default="ws/hetero-proxy/results/kvtier/hf_ref_v3/hf_reference_kv_v3.json")
    ap.add_argument("--seats", type=int, default=96)
    ap.add_argument("--slots", type=int, default=24)
    ap.add_argument("--think", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=180.0)
    ap.add_argument("--duration", type=float, default=900.0)
    ap.add_argument("--max_b", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="r1")
    args = ap.parse_args()
    rank, world, dev = init()
    pin(rank)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    users = [u for u in json.load(open(args.users))["users"] if u["total"] <= MAX_TOTAL]
    n_slots = args.seats if args.policy == "ideal" else args.slots
    if args.mode == "check":
        n_slots = 8
    m = K2Engine(rank, world, dev, n_slots, MAX_LEN, args.max_b)
    host = HostKV(m, args.seats if args.mode == "run" else 1, MAX_TOTAL) \
        if (args.policy == "tier" or args.mode == "check") else None
    gloo = dist.new_group(backend="gloo")
    dist.barrier()
    if rank == 0:
        print(f"[load] {time.time() - t0:.0f} s; slots {n_slots}, host regions "
              f"{(host.nbytes / 2**30) if host else 0:.1f} GiB per rank", flush=True)
    t1 = time.time()
    m.capture_all()
    dist.barrier()
    if rank == 0:
        print(f"[capture] {len(m.gs)} graphs in {time.time() - t1:.0f} s, GPU mem "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    ok = False
    try:
        if args.mode == "check":
            check(m, host, rank, args)
        else:
            serve(m, host, rank, args, users, gloo)
        ok = True
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if ok:
            dist.barrier()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
