#!/bin/bash
# MoE part 2 (after moe1): correct C++ expert-only arms (overrides need a quantized main type), and real size-matched
# mixes = experts (C++ or Python GPTQ) + Q8_0 for every other weight + Q6_K embd/output, comparable to Unsloth UD.
M=$HOME/macaq; cd $M
LB=$M/src/llamacpp/llama.cpp/build/bin; FB=$HOME/rocmfpx-hf/fork-raon/build-cpu/bin
GP=$M/src/llamacpp/llama.cpp/gguf-py; PY=$M/venv/bin/python; WIKI=$HOME/rocmfpx-hf/calib/wiki.test.raw
W=$M/work/moe1; L=$M/logs/moe1; R=$L/results.tsv
export LLAMA_ROCMFPX_TENSOR_SCALE=1
say(){ echo "$(date +%m-%d\ %H:%M:%S) $*" >> $M/logs/steps.log; }
PPL="-f $WIKI -c 512 --chunks 40"
kld(){ grep -m1 "^Mean    KLD" $1 | awk '{print $3}'; }
top(){ grep -m1 "^Same top p" $1 | awk '{print $4}'; }
row(){ printf "%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" >> $R; say "MOE2 $1 bytes=$2 KLD=$3 top=$4"; }
score(){ # name quantized-gguf [bytes]
  $FB/llama-quantize --allow-requantize $2 $W/dq.gguf F16 16 > /dev/null 2>&1 && $PY -I $M/macaq_unrot.py $GP $W/dq.gguf $W/un.gguf > $L/unrot-$1.log 2>&1
  $LB/llama-perplexity -m $W/un.gguf $PPL -ngl 99 --kl-divergence-base $W/base.bin --kl-divergence > $L/kld-$1.log 2>&1 < /dev/null
  row $1 "${3:-$(stat -c %s $2)}" "$(kld $L/kld-$1.log)" "$(top $L/kld-$1.log)"
  rm -f $W/dq.gguf $W/un.gguf
}
EX='ffn_(gate|up|down)_exps'
until grep -q MOE1_DONE $M/logs/steps.log; do sleep 60; done
say MOE2_START
QA="--imatrix $W/im.gguf --pure --token-embedding-type q6_K --output-tensor-type q6_K"
# 1) experts only, rest F16
for t in Q2_0_ROCMFPX Q3_0_ROCMFPX; do
  q=$W/c-$t.gguf
  $FB/llama-quantize $QA --tensor-type "$EX=$t" --tensor-type '\.weight=f16' $W/f16.gguf $q $t 16 > $L/c-$t.log 2>&1 || { say "MOE2 QFAIL $t"; continue; }
  say "MOE2 $t expert tensors converted: $(grep -c "type = *$(echo $t | tr A-Z a-z)" $L/c-$t.log)"
  score cppfix_$t $q; rm -f $q
done
# 2) rest Q8_0 only (experts F16): the non-expert share of the error
$FB/llama-quantize $QA --tensor-type "$EX=f16" $W/f16.gguf $W/r8.gguf Q8_0 16 > $L/r8.log 2>&1 && score rest_q8_only $W/r8.gguf && rm -f $W/r8.gguf
# 3) real shippable mix: C++ S40 experts + Q8_0 rest (true file size)
$FB/llama-quantize $QA --tensor-type "$EX=q2_0_rocmfpx" $W/f16.gguf $W/mix-cpp.gguf Q8_0 16 > $L/mix-cpp.log 2>&1 && score mix_cpp_S40_q8rest $W/mix-cpp.gguf
B=$(stat -c %s $W/mix-cpp.gguf 2>/dev/null); rm -f $W/mix-cpp.gguf
# 4) same mix with Python GPTQ experts (experts kept exactly as exported, rest quantized like 3); size = same types as 3
for k in $W/keep-*.gguf; do
  n=$(basename $k .gguf); n=${n#keep-}
  $FB/llama-quantize $QA --tensor-type "$EX=f16" --allow-requantize $k $W/mx.gguf Q8_0 16 > $L/mix-$n.log 2>&1 && score mix_py_${n}_q8rest $W/mx.gguf $B
  rm -f $W/mx.gguf
done
say MOE2_DONE
