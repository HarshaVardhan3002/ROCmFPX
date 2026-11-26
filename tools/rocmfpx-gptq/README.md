# rocmfpx-gptq

Error-correcting (GPTQ-style) encoders for the existing ROCmFPX low-bit formats, plus the tools used to score them.
The output files use the SAME tensor types as `llama-quantize` (Q2_0_ROCMFPX, Q3_0_ROCMFPX, Q4_0_ROCMI4), so no
kernel or runtime change is needed to run them.

## Files

| file | what it does |
|---|---|
| `rq.py` | dense quantizer: bit-exact simulation of the C++ encoders (S40, FP3, I4, tensor scales), GPTQ on top, optional MACaQ transform, flat R2/R3 study formats; `--real` writes a real ROCmFPX GGUF |
| `rq_moe.py` | the same for the routed experts of Qwen3.5/3.6-MoE (batched GPTQ, per-expert Hessians with shrinkage `--n0`, optional `--rw` routing-weight weighting) |
| `macaq_unrot.py` | turns an F16 expansion (tensor-scaled and/or MACaQ-transformed) back into a plain F16 GGUF for stock scoring |
| `selftest.sh` | checks the simulator against `llama-quantize` tensor by tensor |
| `realtest.sh` | checks that `--real` files expand bit for bit to the F16 export |
| `phase1.sh`, `moe1.sh` ... `moe4.sh` | the experiment runs (paths are for the DGX Spark setup) |
| `smoke_moe.py` | builds a tiny random Qwen3.5-MoE for a quick smoke test |

## Results so far (Qwen3-4B, all 7 linears per layer, KLD vs F16, 40 x 512 wikitext)

| format | C++ encoder today | GPTQ (same format) | GPTQ + MACaQ transform |
|---|---|---|---|
| Q2_0_ROCMFPX (S40, 2.5 bpw) | 1.509 | 0.761 | 0.644 |
| Q3_0_ROCMFPX (FP3, 3.5 bpw) | 0.314 | 0.212 | |
| Q4_0_ROCMI4 (I4, 4.25 bpw) | 0.095 | 0.067 | 0.062 |

For reference IQ2_XXS = 1.512, IQ4_XS = 0.070. The MACaQ column needs an online activation transform that the
runtime does not have yet; the GPTQ column runs on today's kernels.

## MoE results (Qwen3.6-35B-A3B, KLD vs F16, 40 x 512 wikitext)

Only the routed experts use the 2-bit S40 format (Q2_0_ROCMFPX); every other weight is Q8_0 and embedding/output are
Q6_K, so the file sizes are real and comparable. Unsloth's dynamic quants are measured the same way.

| build | size | KLD | same top token |
|---|---|---|---|
| S40 experts, C++ encoder today | 12.79 GB | 0.1724 | 82.9 % |
| S40 experts, GPTQ, n0 = 0 (no shrinkage) | 12.79 GB | 0.2359 | 80.9 % |
| S40 experts, GPTQ, n0 = 512 | 12.79 GB | 0.1804 | 82.8 % |
| S40 experts, GPTQ, n0 = 2048 | 12.79 GB | 0.1688 | 83.0 % |
| S40 experts, GPTQ, n0 = 2048, `--rw` | 12.79 GB | 0.1716 | 82.6 % |
| S40 experts, GPTQ, n0 = 8192 | 12.79 GB | 0.1650 | 83.2 % |
| Unsloth UD-IQ1_M | 10.05 GB | 0.1876 | 81.7 % |
| Unsloth UD-IQ2_XXS | 10.76 GB | 0.1561 | 83.1 % |
| Unsloth UD-IQ2_M | 11.52 GB | 0.1374 | 84.0 % |
| Unsloth UD-Q2_K_XL | 12.29 GB | 0.1089 | 86.0 % |
| Unsloth UD-IQ3_XXS | 13.21 GB | 0.0844 | 88.0 % |

Honest reading: on this MoE model GPTQ gives only a small gain over the C++ encoder (0.172 -> 0.165), and **every one
of our builds is beaten by a smaller Unsloth file**: UD-IQ2_XXS is 2 GB smaller and still better; at about the same size UD-Q2_K_XL is far better.
Per-expert Hessians need strong shrinkage toward the layer Hessian (n0 = 0 is worse than no GPTQ at all), because most
experts see few calibration tokens. The other weights are not the problem: with only the non-expert weights at Q8_0
(experts F16) KLD is 0.0049. The gap comes from spending the same 2.5 bits on every expert; Unsloth mixes bit widths
per tensor. A mixed-width expert layout is the next thing to try, not more calibration.
