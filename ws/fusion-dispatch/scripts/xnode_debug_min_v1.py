# fusion-dispatch E4 debug: minimal 2-node NCCL program with per-step prints and a traceback dump after 40 s
# (used to find where the first cross-node collective hangs). Launch with run_xnode.sh.
import faulthandler
import os
import sys
import time

import torch
import torch.distributed as dist

faulthandler.dump_traceback_later(40, exit=False)
r = os.environ.get("RANK")
def p(m): print(f"[dbg rank {r} {os.uname().nodename} {time.strftime('%T')}] {m}", flush=True)
p("start"); dist.init_process_group("nccl"); p("init_process_group done")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); p(f"device {torch.cuda.current_device()}")
x = torch.ones(1024, device="cuda"); p("tensor")
dist.all_reduce(x); p("all_reduce launched"); torch.cuda.synchronize(); p(f"all_reduce done value {x[0].item()}")
dist.barrier(); p("barrier done")
dist.destroy_process_group(); p("end")
