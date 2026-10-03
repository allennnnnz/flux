# E4 cross-node NCCL hang: debug runs (2026-10-03, all under exclusive_guard_v2, 1 GPU per node)

Symptom: every cross-node NCCL collective hung after "Init COMPLETE" (no error), with IB/GDR, IB without GDR
(../e4_anchor_nccl/w2_B_nogdr) and plain sockets (../e4_anchor_nccl/w2_C_socket).
- min1: per-step prints; both ranks stuck in torch.cuda.synchronize() after the first all_reduce.
- min2_plain: same with only NCCL variables (no Flux / NVSHMEM env) -> still hangs.
- local_shm (158), local_159: single node, 2 GPUs via SHM and via IB loopback with GDRDMA -> both work.
- ib_direction: RDMA 159->158 98.05 Gb/s, bidirectional 193.69 Gb/s, send/recv 98.05 Gb/s -> network fine.
- css-host-159_dot_nccl.conf.txt: 159 has ~/.nccl.conf (NCCL_ALGO=RING, NCCL_PROTO=Simple, NCCL_P2P_LEVEL=NVL,
  NCCL_IB_HCA=mlx5_3:1); NCCL reads it automatically (via the passwd home dir) on 159 only; 158 has none.
- min3_noconf: NCCL_CONF_FILE=/dev/null does NOT stop it in NCCL 2.21.5 (still hangs; min4 logs show
  "NCCL_ALGO set by environment to RING" on 159).
- min4_simple: NCCL_PROTO=Simple on both, ALGO only forced on 159 -> all_reduce "completes" with a WRONG result
  (rank 1 value 0.0, expected 2.0), then hangs. min4_ll: NCCL_PROTO=LL on both -> works.
- min5_same_as_conf: ALGO=Ring, PROTO=Simple, P2P_LEVEL=NVL on both -> works, correct result 2.0.
Cause: the two nodes run NCCL with different algorithm / protocol settings.
