#!/bin/bash
# G3b (11-24): REAL ROCmIQ GGUFs from the GPTQ encoder (rq3 --real, codes packed while quantizing, packer checked
# byte-exact vs llama-quantize), Qwen3-4B, plain (no online transform), scored like G3a: real file -> F16 expansion by
# the fork -> undo TS -> KLD vs F16 (40 x 512 wiki). Also scores rq3 F16 output (sim) of the same run.
M=$HOME/macaq; cd $M
LB=$M/src/llamacpp/llama.cpp/build/bin; FB=$HOME/sub2/build-cpu/bin; GP=$HOME/sub2/gguf-py; PY=$M/venv/bin/python
CAL=$HOME/rocmfpx-hf/calib/calibration_datav3.txt; WIKI=$HOME/rocmfpx-hf/calib/wiki.test.raw
F16=$M/models/qwen3-4b-f16.gguf; HF=$M/models/Qwen3-4B; G=$M/work/go1; HDR=$HOME/sub2/ggml/src/ggml-common.h
W=$M/work/g3b; L=$M/logs/g3b; mkdir -p $W $L; R=$L/results.tsv
say(){ echo "$(date +%m-%d\ %H:%M:%S) $*" >> $M/logs/steps.log; }
kld(){ $LB/llama-perplexity -m $1 -f $WIKI -c 512 --chunks 40 -ngl 99 --kl-divergence-base $G/base.bin --kl-divergence > $2 2>&1 < /dev/null
  echo "$(grep -m1 "^Mean    KLD" $2 | awk "{print \$3}") $(grep -m1 "^Same top p" $2 | awk "{print \$4}")"; }
say G3B_START
[ -f $R ] || printf "fmt\ttype\tenc\tfile\tbytes\tKLD\tsame_top_p\tmin\n" > $R
for spec in "RIQ2 ROCMIQ2_XXS" "RIQ1 ROCMIQ1_S"; do
  set -- $spec; f=$1; t=$2; o=$W/o-$f.gguf; t0=$(date +%s)
  $PY -I $M/rq/rq3.py --gguf-py $GP --hf $HF --f16 $F16 --template $M/work/p1/template.gguf --imatrix $G/im.gguf --calib $CAL \
      --fmt $f --enc gptq --transform none --grid-header $HDR --real $M/work/g3/q-plain-$t.gguf --out $o > $L/rq-$f.log 2>&1 \
      || { say "G3B RQFAIL $f $(tail -1 $L/rq-$f.log)"; continue; }
  mins=$(( ($(date +%s)-t0)/60 ))
  r=$(kld $o $L/kld-sim-$f.log); set -- $r
  printf "%s\t%s\tgptq\tsim-f16\t-\t%s\t%s\t%s\n" $f $t $1 $2 $mins >> $R; say "G3B $f sim KLD=$1 top=$2"
  q=$o.real.gguf; by=$(stat -c %s $q)
  $FB/llama-quantize --allow-requantize $q $W/dq.gguf F16 16 > $L/expand-$f.log 2>&1 || { say "G3B XFAIL $f"; continue; }
  $PY -I $M/macaq_unrot.py $M/src/llamacpp/llama.cpp/gguf-py $W/dq.gguf $W/un.gguf > $L/unrot-$f.log 2>&1 || { say "G3B UFAIL $f"; continue; }
  r=$(kld $W/un.gguf $L/kld-real-$f.log); set -- $r
  printf "%s\t%s\tgptq\treal\t%s\t%s\t%s\t%s\n" $f $t $by $1 $2 $mins >> $R; say "G3B $f REAL $by KLD=$1 top=$2"
  rm -f $W/dq.gguf $W/un.gguf $o
done
say G3B_DONE
