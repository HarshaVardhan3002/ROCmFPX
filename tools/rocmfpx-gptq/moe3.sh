#!/bin/bash
# MoE part 3 (after moe2): GPTQ with routing-weight-weighted expert Hessians (--rw), then its Q8_0-rest mix
M=$HOME/macaq; cd $M
LB=$M/src/llamacpp/llama.cpp/build/bin; FB=$HOME/rocmfpx-hf/fork-raon/build-cpu/bin
GP=$M/src/llamacpp/llama.cpp/gguf-py; PY=$M/venv/bin/python; CAL=$HOME/rocmfpx-hf/calib/calibration_datav3.txt
WIKI=$HOME/rocmfpx-hf/calib/wiki.test.raw; HF=$M/models/qwen36/hf; W=$M/work/moe1; L=$M/logs/moe1; R=$L/results.tsv
export LLAMA_ROCMFPX_TENSOR_SCALE=1
say(){ echo "$(date +%m-%d\ %H:%M:%S) $*" >> $M/logs/steps.log; }
PPL="-f $WIKI -c 512 --chunks 40"
kld(){ grep -m1 "^Mean    KLD" $1 | awk "{print \$3}"; }; top(){ grep -m1 "^Same top p" $1 | awk "{print \$4}"; }
row(){ printf "%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" >> $R; say "MOE3 $1 bytes=$2 KLD=$3 top=$4"; }
until grep -q MOE2_DONE $M/logs/steps.log; do sleep 60; done
say MOE3_START
n=S40-gptq-n2048-rw; o=$W/py-$n.gguf
$PY -I $M/rq/rq_moe.py --gguf-py $GP --hf $HF --template $W/template.gguf --calib $CAL --fmt S40 --enc gptq --n0 2048 --rw \
    --stats $L/stats-$n.json --out $o > $L/rq-$n.log 2>&1 || { say "MOE3 RQFAIL $n"; exit 1; }
$LB/llama-perplexity -m $o $PPL -ngl 99 --kl-divergence-base $W/base.bin --kl-divergence > $L/kld-py-$n.log 2>&1 < /dev/null
row py_$n - "$(kld $L/kld-py-$n.log)" "$(top $L/kld-py-$n.log)"
B=$(awk -F"\t" "\$1==\"mix_cpp_S40_q8rest\"{print \$2}" $R | tail -1)
$FB/llama-quantize --imatrix $W/im.gguf --pure --token-embedding-type q6_K --output-tensor-type q6_K --tensor-type "ffn_(gate|up|down)_exps=f16" $o $W/mx.gguf Q8_0 16 > $L/mix-$n.log 2>&1
$FB/llama-quantize --allow-requantize $W/mx.gguf $W/dq.gguf F16 16 > /dev/null 2>&1 && $PY -I $M/macaq_unrot.py $GP $W/dq.gguf $W/un.gguf > /dev/null 2>&1
$LB/llama-perplexity -m $W/un.gguf $PPL -ngl 99 --kl-divergence-base $W/base.bin --kl-divergence > $L/kld-mix-$n.log 2>&1 < /dev/null
row mix_py_${n}_q8rest "$B" "$(kld $L/kld-mix-$n.log)" "$(top $L/kld-mix-$n.log)"
mv $o $W/keep-$n.gguf; rm -f $W/mx.gguf $W/dq.gguf $W/un.gguf
say MOE3_DONE
