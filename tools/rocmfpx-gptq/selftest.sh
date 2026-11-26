#!/bin/bash
# exactness of the Python format sim vs C++ llama-quantize (diag encoder, same imatrix, TS on)
M=$HOME/macaq; FB=$HOME/rocmfpx-hf/fork-raon/build-cpu/bin; GP=$M/src/llamacpp/llama.cpp/gguf-py; PY=$M/venv/bin/python
W=$M/work/rq; mkdir -p $W; export LLAMA_ROCMFPX_TENSOR_SCALE=1
for t in Q2_0_ROCMFPX:S40 Q3_0_ROCMFPX:FP3 Q4_0_ROCMI4:I4; do
  ct=${t%%:*}; f=${t##*:}
  $FB/llama-quantize --imatrix $M/work/go1/im.gguf --pure --token-embedding-type q6_K --output-tensor-type q6_K $M/models/qwen3-4b-f16.gguf $W/st.gguf $ct 8 > /dev/null 2>&1
  $FB/llama-quantize --allow-requantize $W/st.gguf $W/st-x.gguf F16 8 > /dev/null 2>&1
  $PY -I $M/rq/rq.py --gguf-py $GP --hf x --f16 $M/models/qwen3-4b-f16.gguf --template x --imatrix $M/work/go1/im.gguf --calib x --out x --fmt $f --selftest 1 --cpp $W/st-x.gguf
  rm -f $W/st.gguf $W/st-x.gguf
done
