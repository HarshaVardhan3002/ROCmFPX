#!/bin/bash
# Phase 1 gate run on Qwen3-4B: Python quantizer arms -> F16 export -> stock KLD vs F16 (CUDA), same protocol as go1.
M=$HOME/macaq; cd $M
LB=$M/src/llamacpp/llama.cpp/build/bin; FB=$HOME/rocmfpx-hf/fork-raon/build-cpu/bin
GP=$M/src/llamacpp/llama.cpp/gguf-py; PY=$M/venv/bin/python
CAL=$HOME/rocmfpx-hf/calib/calibration_datav3.txt; WIKI=$HOME/rocmfpx-hf/calib/wiki.test.raw
F16=$M/models/qwen3-4b-f16.gguf; HF=$M/models/Qwen3-4B; G=$M/work/go1
W=$M/work/p1; L=$M/logs/p1; mkdir -p $W $L; R=$L/results.tsv
say(){ echo "$(date +%m-%d\ %H:%M:%S) $*" >> $M/logs/steps.log; }
say P1_START
if [ ! -f $W/template.gguf ]; then
  $FB/llama-quantize --pure --token-embedding-type q6_K --output-tensor-type q6_K $F16 $W/t6.gguf F16 16 > $L/template.log 2>&1
  $FB/llama-quantize --allow-requantize $W/t6.gguf $W/template.gguf F16 16 >> $L/template.log 2>&1 && rm -f $W/t6.gguf
fi
[ -f $R ] || printf "fmt\tenc\ttransform\tKLD\tsame_top_p\n" > $R
arm(){ # fmt enc transform [search]
  local f=$1 e=$2 tr=$3 se=${4:-cpp} o=$W/o.gguf extra=""
  [ $tr = macaq ] && extra="--rot $G/rot-f16.gguf" && im=$G/imr.gguf || im=$G/im.gguf
  $PY -I $M/rq/rq.py --gguf-py $GP --hf $HF --f16 $F16 --template $W/template.gguf --imatrix $im --calib $CAL \
      --fmt $f --enc $e --transform $tr --search $se $extra --out $o > $L/rq-$f-$e-$tr-$se.log 2>&1 || { say "P1 RQFAIL $f $e $tr"; return; }
  $LB/llama-perplexity -m $o -f $WIKI -c 512 --chunks 40 -ngl 99 --kl-divergence-base $G/base.bin --kl-divergence > $L/kld-$f-$e-$tr-$se.log 2>&1 < /dev/null
  local k=$(grep -m1 "^Mean    KLD" $L/kld-$f-$e-$tr-$se.log | awk '{print $3}') t=$(grep -m1 "^Same top p" $L/kld-$f-$e-$tr-$se.log | awk '{print $4}')
  printf "%s\t%s\t%s\t%s\t%s\n" $f $e $tr "$k" "$t" >> $R; say "P1 $f $e $tr $se KLD=$k top=$t"; rm -f $o
}
for f in S40 FP3 I4; do arm $f diag none; arm $f gptq none; done
arm I4 diag none exhaustive; arm I4 gptq none exhaustive
for f in R2 R3; do arm $f diag none; arm $f gptq none; done
arm S40 gptq macaq; arm I4 gptq macaq; arm R2 gptq macaq
say P1_DONE
