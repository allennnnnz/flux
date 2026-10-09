# I4 dry run (2026-10-09 08:12-08:14Z) — NOT a measurement

3 rounds, no guard, run BEFORE the I4 script was committed and BEFORE the official run, with two bugs since fixed:
double cudaHostRegister (sticky error 712; hxb/cudamem.py now keeps a registry) and the correctness reference copying
GPU0 -> GPU4 directly (80 MiB over NVLink on GPU0 / GPU4; hxb/pipeline.py reference() now hops through the CPU).
It already suggested that pre-registered claim P2 (shared copy queue faster than separate for n >= 4) would fail
for pemu4. The claims in scripts/i4_proxy_v1.py were NOT changed after this run.
