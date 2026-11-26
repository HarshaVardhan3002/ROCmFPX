#!/bin/bash
# G3a: real ROCmIQ GGUFs from the C encoder (imatrix only, no GPTQ) on Qwen3-4B, scored exactly like go1:
# quantize -> F16 expansion -> undo TS / transform -> KLD vs F16 (40 x 512 wiki).
M=$HOME/macaq; cd $M
LB=$M/src/llamacpp/llama.cpp/build/bin
FB=$HOME/sub2/build-cpu/bin
GP=$M/src/llamacpp/llama.cpp/gguf-py
PY=$M/venv/bin/python
WIKI=$HOME/rocmfpx-hf/calib/wiki.test.raw
F16=$M/models/qwen3-4b-f16.gguf
G=$M/work/go1
W=$M/work/g3; mkdir -p $W $M/logs/g3
L=$M/logs/g3; R=$L/results.tsv; T=16
export LLAMA_ROCMFPX_TENSOR_SCALE=1
say(){ echo "$(date +%m-%d\ %H:%M:%S) $*" >> $M/logs/steps.log; }
say G3A_START
[ -f $R ] || printf "arm\ttype\tbytes\tKLD\tsame_top_p\tPPL\tquant_s\n" > $R
score(){ # arm type src imatrix
  local a=$1 t=$2 src=$3 im=$4 q=$W/q-$1-$2.gguf dq=$W/dq.gguf un=$W/un.gguf
  local t0=$(date +%s)
  $FB/llama-quantize --imatrix $im --pure --token-embedding-type q6_K --output-tensor-type q6_K $src $q $t $T > $L/quant-$a-$t.log 2>&1 || { say "G3 QFAIL $a $t"; return; }
  local qs=$(( $(date +%s) - t0 ))
  local by=$(stat -c %s $q)
  $FB/llama-quantize --allow-requantize $q $dq F16 $T > $L/expand-$a-$t.log 2>&1 || { say "G3 XFAIL $a $t"; rm -f $q; return; }
  $PY -I $M/macaq_unrot.py $GP $dq $un > $L/unrot-$a-$t.log 2>&1 || { say "G3 UFAIL $a $t"; rm -f $q $dq; return; }
  $LB/llama-perplexity -m $un -f $WIKI -c 512 --chunks 40 -ngl 99 --kl-divergence-base $G/base.bin --kl-divergence > $L/kld-$a-$t.log 2>&1 < /dev/null
  local kld=$(grep -m1 "^Mean    KLD" $L/kld-$a-$t.log | awk '{print $3}')
  local top=$(grep -m1 "^Same top p" $L/kld-$a-$t.log | awk '{print $4}')
  local ppl=$(grep -m1 "^Mean PPL(Q)" $L/kld-$a-$t.log | awk '{print $3}')
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $a $t $by "$kld" "$top" "$ppl" $qs >> $R
  say "G3 $a $t $by KLD=$kld top=$top quant=${qs}s"
  rm -f $dq $un
}
for t in ROCMIQ2_XXS ROCMIQ1_S; do
  score plain $t $F16 $G/im.gguf
  score macaq $t $G/rot-f16.gguf $G/imr.gguf
done
say G3A_DONE
