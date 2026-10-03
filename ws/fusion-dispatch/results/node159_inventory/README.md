# css-host-159 read-only inventory (2026-10-03, from css-host-158 session)

No GPU work, no installs, nothing changed on css-host-159. Files:
01 GPUs / driver / compute apps on 159; 02 nvidia-smi topo -m on 159; 03 RDMA NICs (link layer, rate, netdev) and
up interfaces on both nodes; 04 ping over the 8 RoCE rails from 158; 05 repo / env / users on 159 (incl. the custom
NCCL in ~/.bashrc); 06 local Flux build flags (NVSHMEM) and environment sizes on 158.
