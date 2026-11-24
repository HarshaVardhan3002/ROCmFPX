#!/bin/bash
# G1 (11-23): ROCmIQ1_S / ROCmIQ2_XXS candidate layouts vs the stock IQ layouts under the same encoder, Qwen3-4B,
# every linear at the format, embd/output Q6_K, 40 x 512 wiki KLD vs F16 (same protocol as phase 2).
# Refs (go1, llama-quantize): IQ2_XXS 1.51 / rot 1.34 ; IQ1_S 5.04 / rot 5.22 ; IQ1_M 3.27 ; P2 T64 gptq+rot 1.47.
M=$HOME/macaq; cd $M
LB=$M/src/llamacpp/llama.cpp/build/bin; GP=$M/src/llamacpp/llama.cpp/gguf-py; PY=$M/venv/bin/python
CAL=$HOME/rocmfpx-hf/calib/calibration_datav3.txt; WIKI=$HOME/rocmfpx-hf/calib/wiki.test.raw
F16=$M/models/qwen3-4b-f16.gguf; HF=$M/models/Qwen3-4B; G=$M/work/go1; HDR=$HOME/sub2/ggml/src/ggml-common.h
W=$M/work/g1; L=$M/logs/g1; mkdir -p $W $L; R=$L/results.tsv
say(){ echo "$(date +%m-%d\ %H:%M:%S) $*" >> $M/logs/steps.log; }
say G1_START
[ -f $R ] || printf "fmt\tbpw\tenc\ttransform\tKLD\tsame_top_p\tmin\n" > $R
arm(){ # fmt enc transform
  local f=$1 e=$2 tr=$3 o=$W/o.gguf extra="" im=$G/im.gguf t0=$(date +%s)
  [ $tr = macaq ] && extra="--rot $G/rot-f16.gguf" && im=$G/imr.gguf
  $PY -I $M/rq/rq3.py --gguf-py $GP --hf $HF --f16 $F16 --template $M/work/p1/template.gguf --imatrix $im --calib $CAL \
      --fmt $f --enc $e --transform $tr $extra --grid-header $HDR --out $o > $L/rq-$f-$e-$tr.log 2>&1 || { say "G1 RQFAIL $f $e $tr $(tail -1 $L/rq-$f-$e-$tr.log)"; return; }
  $LB/llama-perplexity -m $o -f $WIKI -c 512 --chunks 40 -ngl 99 --kl-divergence-base $G/base.bin --kl-divergence > $L/kld-$f-$e-$tr.log 2>&1 < /dev/null
  local k=$(grep -m1 "^Mean    KLD" $L/kld-$f-$e-$tr.log | awk '{print $3}') t=$(grep -m1 "^Same top p" $L/kld-$f-$e-$tr.log | awk '{print $4}')
  local b=$($PY -I -c "import sys; sys.path.insert(0,'$M/rq'); import rq3; print(rq3.BPW['$f'])")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $f $b $e $tr "$k" "$t" $(( ($(date +%s)-t0)/60 )) >> $R; say "G1 $f ($b bpw) $e $tr KLD=$k top=$t"; rm -f $o
}
arm IQ2W gptq macaq
arm V64X4 gptq macaq
arm IQ2G gptq macaq
arm IQ2W gptq none
arm V64X4 gptq none
arm S40W gptq macaq
arm IQ1G gptq macaq
say G1_DONE
