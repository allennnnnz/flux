#!/bin/bash
# topic search T2 (2026-10-30): single-node RoCE loopback on the GPU0/GPU1 PCIe switch (NIC 1d:00.x = mlx5_0/mlx5_1).
# Question: can a send path that stages through host memory beat the GPUDirect-read cap (~7 GB/s per GPU in E4),
# also while GPU0/GPU1 copy to host (the uplink's upstream direction) at the same time? And is the GDR-read cap per QP?
# Destination is always GPU memory on the same switch (in-switch P2P write, no uplink) or host (case A only).
# Loopback: client and server on the same port; the NIC reads the source and writes the destination itself.
# Usage (inside the exclusive guard, in tmux): t2_rdma_paths_v1.sh <outdir> <rounds>
set -u
cd /home/rogerlee/flux
OUT=$1; R=${2:-5}
mkdir -p "$OUT"
PY="pixi run --manifest-path pixi.toml python ws/hetero-proxy/scripts/topic/t2_copy_bg_v1.py"
IBW="taskset -c 0-31 ib_write_bw -x 3 -s 1048576 -D 4 --report_gbits -F --tclass=104"
port=19000

# one RDMA pair: dev ip src dst q tag   (src/dst: host | gpuN)
pair() {
  local dev=$1 ip=$2 src=$3 dst=$4 q=$5 tag=$6
  port=$((port + 1))
  local so="" co=""
  [[ $dst == gpu* ]] && so="--use_cuda=${dst#gpu}"
  [[ $src == gpu* ]] && co="--use_cuda=${src#gpu}"
  $IBW -d $dev -q $q $so -p $port > "$OUT/${tag}_server.txt" 2>&1 &
  sleep 1.5
  $IBW -d $dev -q $q $co -p $port $ip > "$OUT/${tag}_client.txt" 2>&1 &
}
bw() { awk '/^ *1048576/ {v=$4} END {print v}' "$1"; }

run_case() {
  local c=$1 r=$2 tag="${1}_r${2}" bgpid=""
  case $c in
    E|F|I)
      $PY --gpus 0,1 --mode d2h --seconds 10 --numa 0 > "$OUT/${tag}_bg.txt" 2>&1 &
      bgpid=$!
      # wait until the copies start (script prints after its 1 s start barrier only at the end, so wait for CUDA init)
      sleep 6 ;;
  esac
  case $c in
    A) pair mlx5_0 10.10.1.158 host host 1 "${tag}_p0" ;;
    B) pair mlx5_0 10.10.1.158 gpu0 gpu1 1 "${tag}_p0" ;;
    C) pair mlx5_0 10.10.1.158 gpu0 gpu1 8 "${tag}_p0" ;;
    D|E) pair mlx5_0 10.10.1.158 host gpu1 1 "${tag}_p0" ;;
    F) pair mlx5_0 10.10.1.158 host gpu1 1 "${tag}_p0"; pair mlx5_1 10.10.2.158 host gpu0 1 "${tag}_p1" ;;
    G) pair mlx5_0 10.10.1.158 gpu0 gpu1 1 "${tag}_p0"; pair mlx5_1 10.10.2.158 gpu1 gpu0 1 "${tag}_p1" ;;
    H) pair mlx5_0 10.10.1.158 gpu0 gpu1 8 "${tag}_p0"; pair mlx5_1 10.10.2.158 gpu1 gpu0 8 "${tag}_p1" ;;
    I) ;;
  esac
  wait
  local line="case=$c round=$r"
  for p in p0 p1; do
    [ -f "$OUT/${tag}_${p}_client.txt" ] && line="$line ${p}_Gbps=$(bw "$OUT/${tag}_${p}_client.txt")"
  done
  [ -n "$bgpid" ] && line="$line bg: $(tail -1 "$OUT/${tag}_bg.txt")"
  echo "$line" | tee -a "$OUT/t2_lines.txt"
}

for r in $(seq 1 $R); do
  for c in $(echo A B C D E F G H I | tr ' ' '\n' | shuf --random-source=<(yes $r)); do
    run_case $c $r
  done
done
echo T2DONE
