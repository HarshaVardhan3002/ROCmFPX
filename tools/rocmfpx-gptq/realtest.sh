#!/bin/bash
# real ROCmFPX packing check: rq.py --real must give a file whose fork expansion equals rq's F16 export bit for bit
M=$HOME/macaq; FB=$HOME/rocmfpx-hf/fork-raon/build-cpu/bin; GP=$HOME/rocmfpx-hf/fork-raon/gguf-py; PY=$M/venv/bin/python
W=$M/work/rt; L=$M/logs/rt; mkdir -p $W $L
export LLAMA_ROCMFPX_TENSOR_SCALE=1
say(){ echo "$(date +%m-%d\ %H:%M:%S) $*" >> $M/logs/steps.log; }
for t in Q2_0_ROCMFPX:S40 Q3_0_ROCMFPX:FP3; do
  T=${t%%:*}; F=${t##*:}
  [ -f $W/c-$T.gguf ] || $FB/llama-quantize --pure --imatrix $M/work/go1/im.gguf --token-embedding-type q6_K --output-tensor-type q6_K \
      $M/models/qwen3-4b-f16.gguf $W/c-$T.gguf $T 16 > $L/cq-$T.log 2>&1
  for enc in diag gptq; do
    $PY -I $M/rq/rq.py --gguf-py $GP --hf $M/models/Qwen3-4B --f16 $M/models/qwen3-4b-f16.gguf --template $M/work/p1/template.gguf \
        --imatrix $M/work/go1/im.gguf --calib $HOME/rocmfpx-hf/calib/calibration_datav3.txt --fmt $F --enc $enc --transform none --search cpp \
        --real $W/c-$T.gguf --out $W/o-$F-$enc.gguf > $L/rq-$F-$enc.log 2>&1 || { say "RT FAIL rq $F $enc"; continue; }
    $FB/llama-quantize --allow-requantize $W/o-$F-$enc.gguf.real.gguf $W/x.gguf F16 16 > $L/ex-$F-$enc.log 2>&1
    $PY -I $M/macaq_unrot.py $GP $W/x.gguf $W/xu.gguf > /dev/null 2>&1
    $PY -I - $GP $W/o-$F-$enc.gguf $W/xu.gguf > $L/cmp-$F-$enc.log 2>&1 <<'PYEOF'
import sys; sys.path.insert(0, sys.argv[1]); import gguf, numpy as np
a = {t.name: t for t in gguf.GGUFReader(sys.argv[2]).tensors}; b = {t.name: t for t in gguf.GGUFReader(sys.argv[3]).tensors}
n = bad = 0; worst = 0.0
for k, t in a.items():
    if not k.startswith("blk.") or k.endswith(".scale") or k not in b or "norm" in k: continue
    x = np.asarray(t.data, dtype=np.float32); y = np.asarray(b[k].data, dtype=np.float32).reshape(x.shape)
    n += 1; d = float(np.abs(x - y).max()); worst = max(worst, d / (float(np.abs(x).max()) + 1e-30))
    if not np.array_equal(x, y): bad += 1
print(f"CMP tensors {n} not-bit-equal {bad} worst-rel {worst:.3e}")
PYEOF
    say "RT $F $enc $(grep -h 'REAL written\|TS differs' $L/rq-$F-$enc.log | head -2 | tr '\n' ' ') $(cat $L/cmp-$F-$enc.log | tail -1)"
    mv $W/o-$F-$enc.gguf.real.gguf $W/real-$F-$enc.gguf; rm -f $W/o-$F-$enc.gguf $W/x.gguf $W/xu.gguf
  done
done
say RT_DONE
