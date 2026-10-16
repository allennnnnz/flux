################################################################################
# hetero-proxy KV tier K2 (2026-10-16): multi-user TP8 engine for the serving run (k2_serve_v1.py).
# Qwen2.5-32B-Instruct on 8 A100s (weights / numerics of e2e/qwen_tp_v1.py), KV cache in fixed SLOTS:
# kc / vc [L, n_slots + 1, 1, max_len, 128]; slot n_slots is a dummy row for padded batch rows. Zero-initialised
# (masked attention multiplies stale entries by 0, so they must be finite).
# Every pass is a CUDA graph (results/kvtier/bench_v5: eager launches cost ~30 ms per pass); inputs are read from
# static device buffers, so one graph serves any conversation / slot / history length of its bucket:
#   decode    graph per batch bucket B: rows read (slot, pos) from d_slot / d_pos, the input token from last_tok[slot];
#             slot-indexed RoPE + KV write and split-K attention (k2_kernels.py); writes the argmax back to
#             last_tok[slot] and tok_log[slot, pos + 1]
#   new       graph per token bucket T (P = 0): a new conversation, or the RECOMPUTE of an evicted one; causal SDPA
#             (flash) on the slot's window [0, T); tokens padded at the end (causal: the padding never reaches a real
#             token; its KV lands past the real end and is overwritten by decode)
#   ext       graph per (T bucket, key window S): T new tokens at positions P.. (P from device memory) attending the
#             slot's cached [0, P) + causally to each other: boolean mask built on the GPU, SDPA (mem-efficient,
#             K / V of the window expanded to the 5 q heads); waits per layer for the hxb gate (restore, as K1)
#             and resets the gates at its end
#   The first generated token goes to last_tok[slot] and tok_log[slot, P + T_real].
# Graph all-reduces use their own communicator (kvtier_v5.py: graph + eager NCCL on one communicator hangs) and
# static output buffers (kvtier_v4.py: NCCL tensors are not reused inside a capture).
# HostKV: this rank's host-memory KV regions (one per user seat), pinned, NUMA-local to the GPU (kvtier_v2.KVTier);
# save = D2H of a token range of a slot, restore = H2D layer by layer + gate writes (kvtier_v5.GRestoreJob).
################################################################################
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "e2e"))
sys.path.insert(0, os.path.join(HERE, ".."))
from qwen_tp_v1 import QwenTP  # noqa: E402
from fused_kernels import add_rmsnorm, silu_mul  # noqa: E402
from k2_kernels import decode_attn_split_slot, rope_cache_slot  # noqa: E402
from hxb import cudamem as cm  # noqa: E402
from kvtier_v2 import node_cpus, numa_pages  # noqa: E402

T_BUCKETS = [16, 32, 64, 128, 256, 512] + list(range(1024, 4097, 512))
B_BUCKETS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
S_STEP = 512


def bucket(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"{n} > largest bucket {buckets[-1]}")


class K2Engine(QwenTP):
    def __init__(self, rank, world, dev, n_slots, max_len, max_b):
        super().__init__(rank, world, dev, n_slots + 1, max_len)
        self.n_slots, self.DUMMY = n_slots, n_slots
        self.kc.zero_()
        self.vc.zero_()
        self.pool = torch.cuda.graph_pool_handle()
        self.pg_graph = dist.new_group(list(range(world)))
        self.gate = torch.zeros(self.L, dtype=torch.int64, device=dev)
        self.b_buckets = [b for b in B_BUCKETS if b <= max_b]
        rows = max(T_BUCKETS[-1], self.b_buckets[-1])
        self.obuf = torch.empty(rows, self.cfg.H, dtype=torch.bfloat16, device=dev)
        self.dbuf = torch.empty(rows, self.cfg.H, dtype=torch.bfloat16, device=dev)
        self.ar = torch.arange(max_len, device=dev)
        # prefill inputs: ids, then [P, slot, last] (last = T_real - 1)
        self.p_ids = torch.zeros(T_BUCKETS[-1], dtype=torch.long, device=dev)
        self.p_meta = torch.tensor([0, self.DUMMY, 0], dtype=torch.long, device=dev)
        self.p_ids_h = torch.zeros(T_BUCKETS[-1], dtype=torch.long).pin_memory()
        self.p_meta_h = torch.zeros(3, dtype=torch.long).pin_memory()
        # decode inputs: [slot rows | pos rows]
        mb = self.b_buckets[-1]
        self.d_in = torch.zeros(2, mb, dtype=torch.long, device=dev)
        self.d_in_h = torch.zeros(2, mb, dtype=torch.long).pin_memory()
        self.last_tok = torch.zeros(n_slots + 1, dtype=torch.long, device=dev)
        self.tok_log = torch.zeros(n_slots + 1, max_len, dtype=torch.long, device=dev)
        self.scr = {}
        self.gs = {}

    def gate_addr(self, i):
        return self.gate.data_ptr() + 8 * i

    # ------------------------------------------------------------------ graph bodies
    def _ar(self, t):
        dist.all_reduce(t, group=self.pg_graph)

    def _mlp_out(self, i, x, a, T):
        c, w = self.cfg, self.layers[i]
        o = torch.mm(a, w["wo"].t(), out=self.obuf[:T])
        self._ar(o)
        x, h2 = add_rmsnorm(x, w["ln2"], c.eps, r=o)
        d = torch.mm(silu_mul(F.linear(h2, w["wgu"])), w["wd"].t(), out=self.dbuf[:T])
        self._ar(d)
        nw = self.layers[i + 1]["ln1"] if i + 1 < self.L else self.norm
        return add_rmsnorm(x, nw, c.eps, r=d)

    def _prefill(self, T, S, gated):
        """T padded new tokens of the sequence in slot p_meta[1] at positions p_meta[0]..; S None: P = 0 (causal
        window [0, T)); else key window [0, S) with a mask from P."""
        c = self.cfg
        hd, nq, nkv = c.HD, self.nq, self.nkv
        P, slot1, last1 = self.p_meta[0:1], self.p_meta[1:2], self.p_meta[2:3]
        pos = P + self.ar[:T]
        slot = slot1.expand(T).contiguous()
        x = F.embedding(self.p_ids[:T], self.embed)
        _, h = add_rmsnorm(x, self.layers[0]["ln1"], c.eps)
        if S is not None:
            mask = (self.ar[:S][None, :] <= pos[:, None])[None, None]
        cs = torch.cuda.current_stream()
        for i in range(self.L):
            w = self.layers[i]
            qkv = F.linear(h, w["wqkv"], w["bqkv"])
            q = rope_cache_slot(qkv, self.cos, self.sin, pos, slot, self.kc[i], self.vc[i], nq, nkv, hd)
            if gated:
                cm.wait_geq(cs, self.gate_addr(i), 1)
            q4 = q.view(1, T, nq, hd).transpose(1, 2)
            if S is None:
                kw = self.kc[i, :, :, :T].index_select(0, slot1)
                vw = self.vc[i, :, :, :T].index_select(0, slot1)
                a = F.scaled_dot_product_attention(q4, kw, vw, is_causal=True, enable_gqa=True, scale=1 / math.sqrt(hd))
            else:
                g = nq // nkv
                kw = self.kc[i, :, :, :S].index_select(0, slot1).repeat_interleave(g, dim=1)
                vw = self.vc[i, :, :, :S].index_select(0, slot1).repeat_interleave(g, dim=1)
                a = F.scaled_dot_product_attention(q4, kw, vw, attn_mask=mask, scale=1 / math.sqrt(hd))
            x, h = self._mlp_out(i, x, a.transpose(1, 2).reshape(T, nq * hd), T)
        tok = F.linear(h.index_select(0, last1), self.lm_head).float().argmax(-1)
        self.last_tok.index_copy_(0, slot1, tok)
        self.tok_log.view(-1).index_copy_(0, slot1 * self.max_len + P + last1 + 1, tok)
        if gated:
            for i in range(self.L):
                cm.write64(cs, self.gate_addr(i), 0)

    def _decode(self, B):
        c = self.cfg
        hd, nq, nkv = c.HD, self.nq, self.nkv
        slot, pos = self.d_in[0, :B], self.d_in[1, :B]
        if B not in self.scr:
            n = B * nkv * 8 * 16
            self.scr[B] = (torch.empty(n, device=self.dev), torch.empty(n, device=self.dev),
                           torch.empty(n * hd, device=self.dev))
        x = F.embedding(self.last_tok.index_select(0, slot), self.embed)
        _, h = add_rmsnorm(x, self.layers[0]["ln1"], c.eps)
        for i in range(self.L):
            w = self.layers[i]
            qkv = F.linear(h, w["wqkv"], w["bqkv"])
            q = rope_cache_slot(qkv, self.cos, self.sin, pos, slot, self.kc[i], self.vc[i], nq, nkv, hd)
            a = decode_attn_split_slot(q, self.kc[i], self.vc[i], pos, slot, 1 / math.sqrt(hd), scratch=self.scr[B])
            x, h = self._mlp_out(i, x, a, B)
        nxt = F.linear(h, self.lm_head).float().argmax(-1)
        self.last_tok.index_copy_(0, slot, nxt)
        self.tok_log.view(-1).index_copy_(0, slot * self.max_len + pos + 1, nxt)

    # ------------------------------------------------------------------ capture
    @torch.inference_mode()
    def _capture(self, key, fn, gated):
        cur = torch.cuda.current_stream()
        s = torch.cuda.Stream()
        s.wait_stream(cur)
        with torch.cuda.stream(s):
            for _ in range(3):
                if gated:
                    self.gate.fill_(1)
                fn()
        cur.wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            fn()
        torch.cuda.synchronize()
        self.gs[key] = g

    @torch.inference_mode()
    def capture_all(self, max_ext_t=T_BUCKETS[-1]):
        """Every decode / new / ext graph. Warm-ups write the dummy slot only (inputs point at it)."""
        self.p_meta.copy_(torch.tensor([0, self.DUMMY, 0]))
        self.d_in[0].fill_(self.DUMMY)
        self.d_in[1].zero_()
        for B in self.b_buckets:
            self._capture(("dec", B), lambda B=B: self._decode(B), False)
        for T in T_BUCKETS:
            if T <= self.max_len:
                self._capture(("new", T), lambda T=T: self._prefill(T, None, False), False)
        for S in range(S_STEP, self.max_len + 1, S_STEP):
            for T in T_BUCKETS:
                if T <= S and T <= max_ext_t:
                    self._capture(("ext", T, S), lambda T=T, S=S: self._prefill(T, S, True), True)
        self.gate.zero_()
        self.kc[:, self.DUMMY].zero_()
        self.vc[:, self.DUMMY].zero_()
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ run
    def set_prefill(self, ids, P, slot):
        T = len(ids)
        self.p_ids_h[:T] = torch.tensor(ids, dtype=torch.long)
        self.p_meta_h[0], self.p_meta_h[1], self.p_meta_h[2] = P, slot, T - 1
        self.p_ids[:T].copy_(self.p_ids_h[:T], non_blocking=True)
        self.p_meta.copy_(self.p_meta_h, non_blocking=True)

    def ext_window(self, P, T):
        return min(self.max_len, -(-(P + bucket(T, T_BUCKETS)) // S_STEP) * S_STEP)

    @torch.inference_mode()  # graphs captured in inference mode must be replayed in it
    def run_new(self, ids, slot):
        """New conversation or recompute: all of ids from position 0."""
        self.set_prefill(ids, 0, slot)
        self.gs[("new", bucket(len(ids), T_BUCKETS))].replay()

    @torch.inference_mode()  # graphs captured in inference mode must be replayed in it
    def run_ext(self, ids, P, slot, gates_ready):
        """ids at positions P.. of slot; gates_ready: no restore in flight (set every gate first)."""
        T = bucket(len(ids), T_BUCKETS)
        assert P + T <= self.max_len, (P, T)
        self.set_prefill(ids, P, slot)
        if gates_ready:
            self.gate.fill_(1)
        self.gs[("ext", T, self.ext_window(P, len(ids)))].replay()

    @torch.inference_mode()  # graphs captured in inference mode must be replayed in it
    def run_decode(self, rows):
        """rows: [(slot, pos)] of the running sequences; padded to a bucket with the dummy slot."""
        B = bucket(len(rows), self.b_buckets)
        self.d_in_h[0, :B] = self.DUMMY
        self.d_in_h[1, :B] = 0
        for j, (s, p) in enumerate(rows):
            self.d_in_h[0, j], self.d_in_h[1, j] = s, p
        self.d_in[:, :B].copy_(self.d_in_h[:, :B], non_blocking=True)
        self.gs[("dec", B)].replay()


class HostKV:
    """This rank's host KV regions, one per user seat: [n_regions, L, 2 (K | V), cap, HD] bf16."""

    def __init__(self, m, n_regions, cap):
        assert m.nkv == 1
        self.m, self.L, self.HD, self.cap = m, m.L, m.cfg.HD, cap
        self.node = 0 if m.dev.index < 4 else 1
        old, nt = os.sched_getaffinity(0), torch.get_num_threads()
        os.sched_setaffinity(0, node_cpus(self.node))
        torch.set_num_threads(1)  # first touch on the GPU's NUMA node
        self.buf = torch.zeros((n_regions, self.L, 2, cap, self.HD), dtype=torch.bfloat16)
        os.sched_setaffinity(0, old)
        torch.set_num_threads(nt)
        self.nbytes = self.buf.numel() * 2
        cm.register(self.buf.data_ptr(), self.nbytes)
        self.numa = numa_pages(self.buf.data_ptr())
        self.stream = torch.cuda.Stream(m.dev)
        self.slot_saved = {}  # slot -> event of the last save that read it

    def _h(self, r, i, kv, a):
        s = self.buf.stride()
        return self.buf.data_ptr() + 2 * (r * s[0] + i * s[1] + kv * s[2] + a * s[3])

    def _g(self, i, kv, slot, a):
        t = self.m.kc if kv == 0 else self.m.vc
        return t.data_ptr() + 2 * (i * t.stride(0) + slot * t.stride(1) + a * t.stride(3))

    def save(self, slot, r, a, e):
        """D2H tokens [a, e) of slot into region r (after the work already queued on the compute stream)."""
        s = self.stream
        s.wait_stream(torch.cuda.current_stream())
        n = (e - a) * self.HD * 2
        for i in range(self.L):
            for kv in (0, 1):
                cm.memcpy(s, self._h(r, i, kv, a), self._g(i, kv, slot, a), n)
        ev = torch.cuda.Event()
        ev.record(s)
        self.slot_saved[slot] = ev

    def before_write(self, slot):
        """The compute stream must not overwrite a slot that a queued save is still reading."""
        ev = self.slot_saved.pop(slot, None)
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)

    def restore_begin(self):
        """Call BEFORE launching the gated graph: the copy stream starts after the work queued so far."""
        self.stream.wait_stream(torch.cuda.current_stream())

    def restore_issue(self, r, slot, a, e):
        s = self.stream
        n = (e - a) * self.HD * 2
        for i in range(self.L):
            if n:
                for kv in (0, 1):
                    cm.memcpy(s, self._g(i, kv, slot, a), self._h(r, i, kv, a), n)
            cm.write64(s, self.m.gate_addr(i), 1)
