#!/bin/bash
# Recreate the isolated vLLM environment used by fusion-dispatch (V1-V12, F4). Idempotent.
#   venv:  /home/rogerlee/venvs/vllm085-flux  (OUTSIDE the repo; --system-site-packages so Flux from the
#          pixi env is importable). vLLM 0.8.5.post1 is the last release on torch 2.6.0 (= pixi's PyPI
#          torch 2.6.0+cu124, CXX11 ABI 0), so Flux needs no rebuild.
#   pins:  transformers==4.51.3 (vLLM 0.8.5 breaks with transformers 5.x: Qwen2Tokenizer has no attribute
#          all_special_tokens_extended); py-spy for stack dumps.
#   extra: Qwen2.5 tokenizer (Apache-2.0, vocab 151,665 >= Llama-3's 128,256) for dummy-weight benchmarks,
#          because the V1 engine loads a tokenizer even with --skip-tokenizer-init (v1/engine/llm_engine.py:77).
# Run from repo root: bash ws/fusion-dispatch/scripts/setup_vllm_venv.sh
set -e
PY=/home/rogerlee/flux/.pixi/envs/default/bin/python
VENV=/home/rogerlee/venvs/vllm085-flux
[ -x $VENV/bin/python ] || uv venv --python $PY --system-site-packages $VENV
uv pip install --python $VENV/bin/python "vllm==0.8.5.post1" "transformers==4.51.3" py-spy
TOK=ws/fusion-dispatch/models/qwen25-tokenizer
mkdir -p $TOK
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  [ -s $TOK/$f ] || curl -sfL -o $TOK/$f https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/$f
done
LD_LIBRARY_PATH=/home/rogerlee/flux/build/lib:/home/rogerlee/flux/python/flux/lib:/home/rogerlee/flux/.pixi/envs/default/lib \
  $VENV/bin/python -c "import torch, vllm, flux, transformers; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'flux ok')"
