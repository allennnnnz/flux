#!/bin/bash
# fusion-dispatch E4 anchor: GPUDirect RDMA bandwidth css-host-158 -> css-host-159 with perftest (--use_cuda),
# NIC mlx5_0 (rail 1), GPU 0 on both sides, 1 MiB messages, 5 s. Cases: host->host, gpu->gpu,
# host->gpu (NIC writes GPU memory on 159), gpu->host (NIC reads GPU memory on 158), plus GPU 1 / mlx5_1.
# Usage: bash ws/fusion-dispatch/scripts/xnode_gdr_anchor_v1.sh <out_dir>   (inside exclusive_guard_v2)
OUT=$1; R=rogerlee@10.2.131.159; OPT="-x 3 -s 1048576 -D 5 --report_gbits -F"
run() {  # name dev subnet client_cuda server_cuda port
  local name=$1 dev=$2 net=$3 cc=$4 sc=$5 port=$6
  ssh -o BatchMode=yes $R "ib_write_bw -d $dev $OPT $sc -p $port" > $OUT/server_$name.txt 2>&1 &
  sleep 3
  ib_write_bw -d $dev $OPT $cc -p $port 10.10.$net.159 > $OUT/client_$name.txt 2>&1
  wait
  echo "$name $(awk '/^ *1048576/ {print $4}' $OUT/client_$name.txt | tail -1) Gb/s" | tee -a $OUT/gdr_anchor_log.txt
}
date '+%F %T start' | tee $OUT/gdr_anchor_log.txt
run host_to_host     mlx5_0 1 ""            ""            18701
run gpu0_to_gpu0     mlx5_0 1 "--use_cuda=0" "--use_cuda=0" 18702
run host_to_gpu0     mlx5_0 1 ""            "--use_cuda=0" 18703
run gpu0_to_host     mlx5_0 1 "--use_cuda=0" ""            18704
run gpu1_to_gpu1_m1  mlx5_1 2 "--use_cuda=1" "--use_cuda=1" 18705
run gpu0_to_gpu0_m1  mlx5_1 2 "--use_cuda=0" "--use_cuda=0" 18706
date '+%F %T done' | tee -a $OUT/gdr_anchor_log.txt
