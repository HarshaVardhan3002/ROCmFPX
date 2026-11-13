#!/bin/bash
# Phase 1 MoE: Qwen3.6-35B-A3B, routed experts only (rest F16, embd/output Q6_K), KLD vs F16 on CUDA.
# Then Unsloth UD baselines (real files) and a size-matched mix for the best arms.
M=$HOME/macaq; cd $M
LB=$M/src/llamacpp/llama.cpp/build/bin; F=$HOME/rocmfpx-hf/fork-raon; FB=$F/build-cpu/bin
GP=$M/src/llamacpp/llama.cpp/gguf-py; PY=$M/venv/bin/python
CAL=$HOME/rocmfpx-hf/calib/calibration_datav3.txt; WIKI=$HOME/rocmfpx-hf/calib/wiki.test.raw
D=$M/models/qwen36; HF=$D/hf; W=$M/work/moe1; L=$M/logs/moe1; mkdir -p $W $L; R=$L/results.tsv
export LLAMA_ROCMFPX_TENSOR_SCALE=1
say(){ echo "$(date +%m-%d\ %H:%M:%S) $*" >> $M/logs/steps.log; }
PPL="-f $WIKI -c 512 --chunks 40"
kld(){ grep -m1 "^Mean    KLD" $1 | awk '{print $3}'; }
top(){ grep -m1 "^Same top p" $1 | awk '{print $4}'; }
row(){ printf "%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" >> $R; say "MOE1 $1 bytes=$2 KLD=$3 top=$4"; }
until grep -q DL36_DONE $M/logs/steps.log; do grep -q DL36_FAIL $M/logs/steps.log && { say MOE1_ABORT_DL; exit 1; }; sleep 60; done
say MOE1_START
[ -f $R ] || printf "arm\tbytes\tKLD\tsame_top_p\n" > $R
[ -f $W/f16.gguf ] || (cd $F && $PY convert_hf_to_gguf.py $HF --outtype f16 --outfile $W/f16.gguf > $L/convert.log 2>&1) || { say MOE1_FAIL convert; exit 1; }
if [ ! -f $W/template.gguf ]; then
  $FB/llama-quantize --pure --token-embedding-type q6_K --output-tensor-type q6_K $W/f16.gguf $W/t6.gguf F16 16 > $L/template.log 2>&1
  $FB/llama-quantize --allow-requantize $W/t6.gguf $W/template.gguf F16 16 >> $L/template.log 2>&1 && rm -f $W/t6.gguf
fi
P1PID=$(pgrep -f "^bash phase1.sh" | head -1); while [ -n "$P1PID" ] && kill -0 $P1PID 2>/dev/null; do sleep 60; done
grep -q RT_DONE $M/logs/steps.log || bash $M/rq/realtest.sh
say MOE1_GPU_START
[ -f $W/base.bin ] || $LB/llama-perplexity -m $W/f16.gguf $PPL -ngl 99 --kl-divergence-base $W/base.bin > $L/base.log 2>&1 < /dev/null || { say MOE1_FAIL base; exit 1; }
[ -f $W/im.gguf ] || $LB/llama-imatrix -m $W/f16.gguf -f $CAL -o $W/im.gguf -c 512 -b 512 -ub 512 -ngl 99 --chunks 128 --no-ppl --output-format gguf > $L/im.log 2>&1 < /dev/null
row template_only "$(stat -c %s $W/template.gguf)" "$($LB/llama-perplexity -m $W/template.gguf $PPL -ngl 99 --kl-divergence-base $W/base.bin --kl-divergence 2>&1 < /dev/null | tee $L/kld-template.log | grep -m1 '^Mean    KLD' | awk '{print $3}')" "$(top $L/kld-template.log)"
say MOE1_PREP_DONE
# Unsloth UD real files (stock types, CUDA)
for f in UD-IQ1_M UD-IQ2_XXS UD-IQ2_M UD-Q2_K_XL UD-IQ3_XXS; do
  g=$D/ud/Qwen3.6-35B-A3B-$f.gguf
  $LB/llama-perplexity -m $g $PPL -ngl 99 --kl-divergence-base $W/base.bin --kl-divergence > $L/kld-$f.log 2>&1 < /dev/null
  row unsloth_$f "$(stat -c %s $g)" "$(kld $L/kld-$f.log)" "$(top $L/kld-$f.log)"
done
# C++ fork encoder, experts only (plain imatrix), S40 and FP3
for t in Q2_0_ROCMFPX Q3_0_ROCMFPX; do
  q=$W/cpp-$t.gguf
  $FB/llama-quantize --imatrix $W/im.gguf --tensor-type ffn_gate_exps=$t --tensor-type ffn_up_exps=$t --tensor-type ffn_down_exps=$t \
     --token-embedding-type q6_K --output-tensor-type q6_K $W/f16.gguf $q F16 16 > $L/cpp-$t.log 2>&1 || { say "MOE1 CPPFAIL $t"; continue; }
  $FB/llama-quantize --allow-requantize $q $W/dq.gguf F16 16 > /dev/null 2>&1 && $PY -I $M/macaq_unrot.py $GP $W/dq.gguf $W/un.gguf > $L/unrot-$t.log 2>&1
  $LB/llama-perplexity -m $W/un.gguf $PPL -ngl 99 --kl-divergence-base $W/base.bin --kl-divergence > $L/kld-cpp-$t.log 2>&1 < /dev/null
  row cpp_$t "$(stat -c %s $q)" "$(kld $L/kld-cpp-$t.log)" "$(top $L/kld-cpp-$t.log)"
  rm -f $q $W/dq.gguf $W/un.gguf
done
# Python arms
pyarm(){ # name fmt enc n0
  local o=$W/py-$1.gguf
  $PY -I $M/rq/rq_moe.py --gguf-py $GP --hf $HF --template $W/template.gguf --calib $CAL --fmt $2 --enc $3 --n0 $4 \
      --stats $L/stats-$1.json --out $o > $L/rq-$1.log 2>&1 || { say "MOE1 RQFAIL $1"; return; }
  $LB/llama-perplexity -m $o $PPL -ngl 99 --kl-divergence-base $W/base.bin --kl-divergence > $L/kld-py-$1.log 2>&1 < /dev/null
  row py_$1 - "$(kld $L/kld-py-$1.log)" "$(top $L/kld-py-$1.log)"
  mv $o $W/keep-$1.gguf 2>/dev/null; ls $W/keep-*.gguf | head -n -2 | xargs -r rm -f   # keep the last two exports only
}
pyarm S40-diag-n0 S40 diag 0
pyarm S40-gptq-n0 S40 gptq 0
pyarm S40-gptq-n2048 S40 gptq 2048
pyarm S40-gptq-n512 S40 gptq 512
# R2/R3 dropped: dense test showed flat row-scale formats fail without rotation (R2 gptq KLD 3.73 vs S40 gptq 0.76)
pyarm FP3-gptq-n2048 FP3 gptq 2048
say MOE1_DONE
