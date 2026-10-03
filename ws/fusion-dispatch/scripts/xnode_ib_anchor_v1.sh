#!/bin/bash
################################################################################
# fusion-dispatch E4 anchor: RDMA line rate of each RoCE rail css-host-158 -> css-host-159 (ib_write_bw,
# host memory, 1 MiB messages, 5 s), each rail alone and all 7 working rails at once. This is the reference
# for every cross-node bandwidth number (CLAUDE.md 5.1.2). Rail 4 (mlx5_3) is down on css-host-158.
# Run inside exclusive_guard_v2 on css-host-158; a monitor-only guard runs on css-host-159 for the window.
# Usage: bash ws/fusion-dispatch/scripts/xnode_ib_anchor_v1.sh <out_dir>
################################################################################
OUT=$1
mkdir -p "$OUT"
R=rogerlee@10.2.131.159
DEV=(mlx5_0 mlx5_1 mlx5_2 mlx5_4 mlx5_5 mlx5_6 mlx5_7)
NET=(1 2 3 5 6 7 8)
OPT="-x 3 -s 1048576 -D 5 --report_gbits -F -q 2"
run_pair() {  # $1 = index, $2 = tag
  local d=${DEV[$1]} n=${NET[$1]} port=$((18515 + $1))
  ssh -o BatchMode=yes $R "ib_write_bw -d $d $OPT -p $port" > "$OUT/server_$2_$d.txt" 2>&1 &
  local sp=$!
  sleep 2
  ib_write_bw -d $d $OPT -p $port 10.10.$n.159 > "$OUT/client_$2_$d.txt" 2>&1
  wait $sp
}
echo "$(date '+%F %T') single-rail runs" | tee "$OUT/ib_anchor_log.txt"
for i in "${!DEV[@]}"; do run_pair $i single; done
echo "$(date '+%F %T') all rails concurrently" | tee -a "$OUT/ib_anchor_log.txt"
for i in "${!DEV[@]}"; do run_pair $i concurrent & done
wait
echo "$(date '+%F %T') done" | tee -a "$OUT/ib_anchor_log.txt"
for f in "$OUT"/client_*.txt; do
  bw=$(awk '/^ *1048576/ {print $4}' "$f" | tail -1)
  echo "$(basename "$f" .txt) BW_average_Gbps=${bw:-FAILED}" | tee -a "$OUT/ib_anchor_log.txt"
done
