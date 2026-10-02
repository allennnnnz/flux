################################################################################
# fusion-dispatch V11: minimal repro of the V4 hang. With 80 blocks, after CUDA graphs that
# captured ~320 torch.distributed (c10d) NCCL collectives each were replayed and destroyed,
# the next eager dist.all_gather_into_tensor blocked on every rank with idle GPUs
# (py-spy: all ranks in distributed_c10d.all_gather_into_tensor).
# Repro: per "size" step, capture G graphs of N back-to-back collectives (all_gather or
# reduce_scatter of M x K bf16), replay R times, destroy them, then do one eager collective.
# backend=c10d uses torch.distributed; backend=pynccl uses vLLM's PyNcclCommunicator (what vLLM
# captures in its graphs). A watchdog thread reports a step that takes longer than --timeout_s.
# Must run in the vLLM venv (launch_vllm_env.sh).
################################################################################
import argparse
import os
import sys
import threading
import time

import torch
import torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["c10d", "pynccl"], required=True)
ap.add_argument("--n_coll", type=int, default=320, help="collectives per graph")
ap.add_argument("--graphs", type=int, default=6)
ap.add_argument("--Ms", default="64,256,384,512")
ap.add_argument("--K", type=int, default=8192)
ap.add_argument("--timeout_s", type=float, default=120)
a = ap.parse_args()
dist.init_process_group("nccl")
RANK, W = dist.get_rank(), dist.get_world_size()
LOCAL = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(LOCAL)
PYN = None
if a.backend == "pynccl":
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    PYN = PyNcclCommunicator(group=dist.new_group(backend="gloo"), device=torch.device(f"cuda:{LOCAL}"))
state = {"step": "init", "t": time.time()}


def watchdog():
    while True:
        time.sleep(5)
        if time.time() - state["t"] > a.timeout_s:
            print(f"[rank{RANK}] STUCK in step '{state['step']}' for {time.time() - state['t']:.0f}s", file=sys.stderr, flush=True)
            os._exit(3)


threading.Thread(target=watchdog, daemon=True).start()


def step(name):
    state["step"], state["t"] = name, time.time()


def ag(out, x):
    if PYN is None:
        dist.all_gather_into_tensor(out, x)
    else:
        PYN.all_gather(out, x)


for M in [int(x) for x in a.Ms.split(",")]:
    x = torch.randn(M // W, a.K, device="cuda").to(torch.bfloat16)
    out = torch.empty(M, a.K, device="cuda", dtype=torch.bfloat16)
    step(f"M={M} eager before capture")
    ag(out, x)
    torch.cuda.synchronize()
    graphs = []
    for gi in range(a.graphs):
        step(f"M={M} capture graph {gi}")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            ag(out, x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(a.n_coll):
                ag(out, x)
        graphs.append(g)
    step(f"M={M} replay")
    for _ in range(5):
        for g in graphs:
            g.replay()
    torch.cuda.synchronize()
    step(f"M={M} destroy")
    del graphs
    torch.cuda.synchronize()
    step(f"M={M} eager after destroy")
    ag(out, x)
    torch.cuda.synchronize()
    if RANK == 0:
        print(f"[{a.backend}] M={M} ok (graphs={a.graphs} x {a.n_coll} collectives)", file=sys.stderr, flush=True)
if RANK == 0:
    print(f"[{a.backend}] ALL OK", file=sys.stderr, flush=True)
os._exit(0)
