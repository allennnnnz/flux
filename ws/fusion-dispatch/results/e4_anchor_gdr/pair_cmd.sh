ssh -o BatchMode=yes rogerlee@10.2.131.159 "ib_write_bw -d mlx5_0 -x 3 -s 1048576 -D 5 --report_gbits -F --tclass=104 --use_cuda=0 -p 18801" > ws/fusion-dispatch/results/e4_anchor_gdr/server_pair_gpu0_m0.txt 2>&1 &
ssh -o BatchMode=yes rogerlee@10.2.131.159 "ib_write_bw -d mlx5_1 -x 3 -s 1048576 -D 5 --report_gbits -F --tclass=104 --use_cuda=1 -p 18802" > ws/fusion-dispatch/results/e4_anchor_gdr/server_pair_gpu1_m1.txt 2>&1 &
sleep 3
ib_write_bw -d mlx5_0 -x 3 -s 1048576 -D 5 --report_gbits -F --tclass=104 --use_cuda=0 -p 18801 10.10.1.159 > ws/fusion-dispatch/results/e4_anchor_gdr/client_pair_gpu0_m0.txt 2>&1 &
ib_write_bw -d mlx5_1 -x 3 -s 1048576 -D 5 --report_gbits -F --tclass=104 --use_cuda=1 -p 18802 10.10.2.159 > ws/fusion-dispatch/results/e4_anchor_gdr/client_pair_gpu1_m1.txt 2>&1 &
wait
