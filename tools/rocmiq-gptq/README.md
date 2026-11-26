# rocmiq-gptq: GPTQ encoder for ROCmIQ2_XXS and ROCmIQ1_S

Research tools for the two sub-2.5 bpw formats on the `rocmfpx-sub2` branch:

| Type | ggml id | Bits per weight | Block (256 weights) |
|---|---:|---:|---|
| `ROCMIQ2_XXS` | 122 | 2.0625 | 16-bit scale word + 32 codes of 16 bits (iq2xxs grid index, 7 signs + even parity, +/-4 shift) |
| `ROCMIQ1_S` | 123 | 1.5625 | 16-bit scale word + 64 six-bit indices into a fixed 64 x 4 int8 codebook |

The scale word holds a UE4M3 block scale (bits 0-7) and a 2-bit exponent per 64 weights
(sub scale = d * {1, 0.707, 0.5, 0.354}).

`llama-quantize` writes both types with a per-block imatrix search. `rq3.py --enc gptq` adds GPTQ error feedback
(inverse-Hessian updates from a transformers forward pass over calibration text) and, with `--real`, writes a
real GGUF of the same type: the codes are recorded while quantizing and packed into the blocks, so the file is
read by the normal kernels (CPU, Vulkan).

## Results (Qwen3-4B, every linear layer at the type, embedding/output Q6_K, KLD vs F16, 40 x 512 tokens WikiText-2)

| File | Encoder | KLD | Same top |
|---|---|---:|---:|
| stock `IQ2_XXS` | llama-quantize | 1.51 | |
| `ROCMIQ2_XXS` | llama-quantize | 1.192 | 58.4% |
| `ROCMIQ2_XXS` | rq3 GPTQ, real file | **0.752** | 67.2% |
| stock `IQ1_S` | llama-quantize | 5.04 | |
| stock `IQ1_M` (1.75 bpw) | llama-quantize | 3.27 | |
| `ROCMIQ1_S` | llama-quantize | 4.464 | 25.5% |
| `ROCMIQ1_S` | rq3 GPTQ, real file | **1.969** | 45.2% |

The real-file KLD equals the simulated (F16) output of the same run (0.752444 vs 0.752444, 1.9688 vs 1.9694), and
`riq_check.py` shows the packer writes the same bytes as `llama-quantize` when GPTQ is off
(ROCMIQ1_S 100%, ROCMIQ2_XXS 99.99% of bytes; the rest are rounding ties).

## Strix Halo (Radeon 8060S, Vulkan), 4B dense model, llama-quantize encoder

Default mix (`llama-quantize ... ROCMIQ2_XXS` with imatrix, `LLAMA_ROCMFPX_TENSOR_SCALE=1`; attn_v and the first
eighth of ffn_down promoted like the stock IQ2_XXS mix), PPL over 30 x 512 tokens, 8-bit base PPL 7.21:

| File | Size | PPL | pp512 tok/s | tg128 tok/s |
|---|---:|---:|---:|---:|
| stock `IQ2_XXS` mix | 1.33 GB | 96.7 | 1846 | 95.4 |
| `ROCMIQ2_XXS` mix | 1.35 GB | **49.6** | 1734 | 88.7 |
| stock `IQ1_S` mix | 1.16 GB | 18503 | 1893 | 107.6 |
| `ROCMIQ1_S` mix | 1.17 GB | 33049 | 1728 | 106.5 |

The same files give the same PPL on the CPU backend (`--device none`; ROCMIQ2 59.84 vs 59.59 Vulkan, 8 chunks),
but the CPU path is slow. 1.56 bpw is not usable on a 4B model with either encoder; it is meant for large models.

## Files

- `rq3.py`: the quantizer (also simulates the other ROCmFPX low-bit formats). `--fmt RIQ2` = `ROCMIQ2_XXS`,
  `--fmt RIQ1` = `ROCMIQ1_S`. `--real C.gguf` takes a `llama-quantize` file of the same type as the layout template
  and writes `<out>.real.gguf`. Needs `gguf-py` from this branch (types 122 / 123).
- `riq_check.py`: packer check against a `llama-quantize` file (bytes, decode == values).
- `g1.sh`, `g3a.sh`, `g3b.sh`: the run scripts behind the tables (layout sweep, C-encoder files, GPTQ real files).
- `macaq_unrot.py`: undoes the tensor scale / rotation in an F16 expansion for scoring.

Dense models only so far (no MoE expert tensors). The scripts expect the work layout used for these runs
(`$HOME/macaq`, a CPU build of this branch in `$HOME/sub2/build-cpu`).
