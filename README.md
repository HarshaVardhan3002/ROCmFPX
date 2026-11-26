<div align="center">

# ROCmFPX

**llama.cpp for AMD GPUs, with low-bit weight types, Strix Halo kernels and new model support.**<br>
Built on current upstream llama.cpp and tested on AMD Strix Halo (Radeon 8060S, gfx1151) with Vulkan and HIP.

![Backends](https://img.shields.io/badge/Vulkan%20%7C%20HIP%20%7C%20CPU-ED1C24)
![Hardware](https://img.shields.io/badge/tested%20on-Strix%20Halo%20gfx1151-2b3137)
![Weight types](https://img.shields.io/badge/weight%20types-11%20ROCm%20types-2b3137)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20models-Harsha001-ED1C24)](https://huggingface.co/Harsha001)
[![License](https://img.shields.io/badge/license-MIT-2b3137)](LICENSE)

[Quick start](#quick-start) · [Weight types](#weight-types) · [What this fork adds](#what-this-fork-adds) · [Models](#models-built-with-this-fork) · [Tests](#tests) · [Credits](#credits)

</div>

> [!NOTE]
> **The original ROCmFPX and ROCmFP4 formats are not mine.** `Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST`, `Q2_0_ROCMFPX` to `Q8_0_ROCMFPX`, `Q4_0_ROCMI4`, their recipes and their first kernels were written by [charlie12345 (Carlo Pasquale)](https://github.com/charlie12345/ROCmFPX), caf, ciru-ai and the other contributors of that repository. This fork carries their work on a newer llama.cpp and adds the parts listed below.

## Quick start

```sh
git clone https://github.com/HarshaVardhan3002/ROCmFPX
cd ROCmFPX

# Vulkan (recommended on Strix Halo: all the fast kernels below are Vulkan kernels)
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j

# or ROCm / HIP on Strix Halo
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151
cmake --build build --config Release -j
```

Run a model exactly as with llama.cpp:

```sh
build/bin/llama-server -m model.gguf -ngl 99 -fa on
```

Nothing else needs to be installed. The new shaders are compiled into the Vulkan backend, and it picks the right kernel for each weight type at run time.

## Weight types

| Type | Bits per weight | From | Notes |
|---|---|---|---|
| `Q8_0_ROCMFPX` | 8.25 | charlie12345 | |
| `Q6_0_ROCMFPX` | 6.50 | charlie12345 | |
| `Q4_0_ROCMFP4` | 4.50 | charlie12345 | 4-bit codebook, UE4M3 scales |
| `Q4_0_ROCMFP4_FAST` | 4.25 | charlie12345 | single-scale layout, faster |
| `Q4_0_ROCMI4` | 4.25 | charlie12345 | signed 4-bit, no codebook; no Vulkan shaders yet |
| `Q3_0_ROCMFPX` | 3.50 | charlie12345 | |
| `Q2_0_ROCMFPX` | 2.50 | charlie12345 | S40 codebook, two UE4M3 scales |
| `Q2_0_ROCMFPL` | 2.125 | this fork | S40 codes in 256-weight blocks, one UE4M3 scale per 64 weights |
| `ROCMIQ2_XXS` | 2.0625 | this fork | IQ2_XXS grid plus a shift bit and a 16-bit scale word |
| `Q1_0_ROCMFPT` | 1.75 | this fork | ternary codes, one UE4M3 scale per 64 weights |
| `ROCMIQ1_S` | 1.5625 | this fork | 4-weight vectors from a 64-entry int8 codebook |

`llama-quantize --help` lists the full recipes (`_STRIX`, `_LEAN`, `_COHERENT`, `_AGENT`, `_EVEN`, and the `ROCMIQ2_XXS` / `ROCMIQ1_S` default mixes).

## What this fork adds

<table>
<tr>
<td width="50%" valign="top">

### 🧮 ROCmIQ: 1-bit and 2-bit types for AMD

Two new types made for big mixture-of-experts models on Strix Halo, with a GPTQ encoder in [`tools/rocmiq-gptq`](tools/rocmiq-gptq).

KL divergence against the original model (lower is better):

| Model and setup | Stock llama.cpp | ROCmIQ |
|---|---|---|
| Qwen3-4B, all 2-bit | IQ2_XXS 1.51 | **ROCMIQ2_XXS 0.75** |
| Qwen3-4B, all 1-bit | IQ1_S 5.04 | **ROCMIQ1_S 1.97** |
| Qwen3.6-35B-A3B, Unsloth's UD-IQ2_XXS tensor map | Unsloth file 0.159 | **0.146** |

On the MoE model the win is smaller than on the dense one: there the choice of which tensors get more bits matters more than the type.

</td>
<td width="50%" valign="top">

### ⚡ Vulkan kernels for Strix Halo

- **Expert kernel for short rows** (ROCMIQ2_XXS and IQ3_XXS experts): one dispatch does up, gate and the GLU. On Qwen3.6-35B-A3B decode goes from 78.6 to 81–83 tokens per second. From memory the expert ops run at about 140 GB/s; with the weights in cache IQ3_XXS reaches 211–250 GB/s.
- **Per-expert `.scale` read inside the mat-vec:** 1635 → 1395 dispatches per token.
- **Faster ternary `Q1_0_ROCMFPT`:** decodes as fast as `Q2_0_ROCMFPX` with 22% less memory.
- Faster `ROCMIQ2_XXS` matrix multiply (prompt speed 1363 → 1891 tokens per second), IQ1_S / IQ1_M always on the integer-dot path on AMD, and tuned small ops for gfx1151.

Each new path can be turned off for comparison: `GGML_VK_DISABLE_EXPERT_MMVQ=1`, `GGML_VK_DISABLE_MOE_GLU_FUSION=1`, `GGML_VK_DISABLE_FUSION=1`.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📏 Tensor scales for very small weights

Opt in with `LLAMA_ROCMFPX_TENSOR_SCALE=1` when quantizing.

- ROCmFPX block scales are UE4M3 with a fixed range, so models with tiny weights (block maximum around 1e-3) lost most of their precision in every ROCmFPX type.
- `llama-quantize` multiplies such tensors by a power of two (per expert for 3D tensors) and writes the inverse as a `.scale` tensor, which the model graph multiplies back in. A bug where 34 model graphs ignored these scales is fixed.
- NVFP4 checkpoints that keep a per-expert global `.scale` get it folded into the weights before requantizing.

**RecursiveSI/flint at `Q8_0_ROCMFPX`:** KL divergence 3.92 → **0.075**, same top token 33% → **91%**.

</td>
<td width="50%" valign="top">

### 🧩 New model support

Architectures that upstream llama.cpp does not have (or did not have when they were added here):

- **DeepSeek-V4.1-Flash** (`deepseek41`), with a gathered-window prompt path for GPUs without sparse flash attention
- **Hy4-flash** (`hy_v4_flash`, Tencent), with the Engram tables
- **Inkling 2** Small and Mini (`inkling2`), plus expert-statistics and pruning tools
- **Raon-1** and **Raon-1 nano** (`raon1`, `raon1nano`, KRAFTON), with the MTP draft head
- **SenseNova-Loop** (`sensenova_loop`), looped layers with two-stream mHC
- **Kolibri-1.1** (sparse attention, cross-layer KV sharing)
- **Clover MoE**, **LFM3**, **Ripple-65B** (`ricursive`) with DFlash2 drafting
- **MiniMax-M3.1**: MXFP8 dense weights and NVFP4 experts

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🗣️ Speech and other runtimes

- **`llama-haru`**: native Haru-4B full-duplex speech runtime
- **Maya2 native** text to speech (Qwen3 backbone, depth decoder, Mimi codec)
- **Audio8-TTS** text to speech
- **`llama-decide`** for the baselabs decision models (elephant-1, mammoth-1); matches the PyTorch reference to within 0.004 in probability

</td>
<td width="50%" valign="top">

### 🛠️ Quantization tools

- [`tools/rocmfpx-gptq`](tools/rocmfpx-gptq) and [`tools/rocmiq-gptq`](tools/rocmiq-gptq): GPTQ encoders for the ROCm types, writing real GGUF files
- `llama-imatrix --expert-stats`: per-expert routing statistics
- `--tensor-type` overrides also work when the main type is F16 or BF16
- `llama-quantize` keeps MTP and EAGLE draft projections at Q8_0 and counts NextN layers correctly

</td>
</tr>
</table>

### Fixes to the original ROCmFPX port

- Vulkan `Q6_0_ROCMFPX` read 32 one-byte values per block, but the block holds 32 packed 6-bit codes in 24 bytes, so every Vulkan result for this type was wrong. Fixed.
- CPU `Q8_0_ROCMFPX`, `Q6_0_ROCMFPX` and `Q4_0_ROCMFP4` take F32 activations, so the CPU result matches the GPU.
- CUDA: restored the MXFP4 `get_rows` case lost in the merge, and upstream's UE4M3 decode (the old change broke subnormal NVFP4 scales on HIP).

> [!WARNING]
> The CUDA kernels for the ROCm types give wrong results on NVIDIA GB10 (sm_121). Use them on CPU, Vulkan or HIP.

## Models built with this fork

Every quantized file is scored by KL divergence against the original model, and the model cards say plainly where stock llama.cpp types are closer. The download badges are live from Hugging Face and count downloads in the last 30 days.

### Big models that fit a 128 GB Strix Halo

Mixed 1-bit and 2-bit builds of large mixture-of-experts models, sized to leave room for the context in 128 GB of unified memory.

| Model | Original by | What it is | Downloads |
|---|---|---|---|
| [**DeepSeek-V4.1-Flash Uncensored**](https://huggingface.co/Harsha001/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-StrixHalo-GGUF) | deepseek-ai; uncensored by Infatoshi | large MoE, 1-bit and 2-bit mix | <a href="https://huggingface.co/Harsha001/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-StrixHalo-GGUF"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FDeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-StrixHalo-GGUF&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**MiniMax-M3.1-Flash**](https://huggingface.co/Harsha001/MiniMax-M3.1-Flash-StrixHalo-GGUF) | MiniMaxAI | large MoE, expert-pruned, about 97 GB | <a href="https://huggingface.co/Harsha001/MiniMax-M3.1-Flash-StrixHalo-GGUF"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FMiniMax-M3.1-Flash-StrixHalo-GGUF&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**GLM-5.3-Flash Uncensored**](https://huggingface.co/Harsha001/GLM-5.3-Flash-Uncensored-StrixHalo-GGUF) | orcarouter | large MoE, 2-bit, 96 GiB | <a href="https://huggingface.co/Harsha001/GLM-5.3-Flash-Uncensored-StrixHalo-GGUF"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FGLM-5.3-Flash-Uncensored-StrixHalo-GGUF&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Nex-N2.6**](https://huggingface.co/Harsha001/Nex-N2.6-StrixHalo-GGUF) | nex-agi | large MoE, 2-bit, 96 GiB | <a href="https://huggingface.co/Harsha001/Nex-N2.6-StrixHalo-GGUF"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FNex-N2.6-StrixHalo-GGUF&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Inkling 2 Small**](https://huggingface.co/Harsha001/Inkling-2-Small-StrixHalo-GGUF) | thinkingmachines | large MoE, 1-bit | <a href="https://huggingface.co/Harsha001/Inkling-2-Small-StrixHalo-GGUF"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FInkling-2-Small-StrixHalo-GGUF&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |

### Language and vision models

| Model | Original by | What it is | Downloads |
|---|---|---|---|
| [**Hy4-flash**](https://huggingface.co/Harsha001/Hy4-flash-ROCmFPX) | Tencent | MoE with Engram tables | <a href="https://huggingface.co/Harsha001/Hy4-flash-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FHy4-flash-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Inkling 2 Mini Uncensored**](https://huggingface.co/Harsha001/Inkling-2-Mini-Uncensored-ROCmFPX) | thinkingmachines | looped core MoE | <a href="https://huggingface.co/Harsha001/Inkling-2-Mini-Uncensored-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FInkling-2-Mini-Uncensored-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Ripple-65B abliterated**](https://huggingface.co/Harsha001/Ripple-65B-abliterated-ROCmFPX) | orcarouter | 65B, with a DFlash draft model (gated) | <a href="https://huggingface.co/Harsha001/Ripple-65B-abliterated-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FRipple-65B-abliterated-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Ornith 1.5 122B**](https://huggingface.co/Harsha001/Ornith-1.5-122B-ROCmFPX) | ornith-ai | 122B, text generation | <a href="https://huggingface.co/Harsha001/Ornith-1.5-122B-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FOrnith-1.5-122B-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**MiniCPM-V 4.7**](https://huggingface.co/Harsha001/MiniCPM-V-4.7-35B-A3B-ROCmFPX) | openbmb | 35B MoE, 3B active | <a href="https://huggingface.co/Harsha001/MiniCPM-V-4.7-35B-A3B-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FMiniCPM-V-4.7-35B-A3B-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Clover-35B-A3B Preview**](https://huggingface.co/Harsha001/Clover-35B-A3B-Preview-ROCmFPX) | Diffuspace | 35B MoE, 3B active | <a href="https://huggingface.co/Harsha001/Clover-35B-A3B-Preview-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FClover-35B-A3B-Preview-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Groot-19B-A3B**](https://huggingface.co/Harsha001/Groot-19B-A3B-261023-ROCmFPX) | GrootTeam | 19B MoE, image and text | <a href="https://huggingface.co/Harsha001/Groot-19B-A3B-261023-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FGroot-19B-A3B-261023-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Nanbeige4.3 33B**](https://huggingface.co/Harsha001/Nanbeige4.3-33B-ROCmFPX) | Nanbeige | 33B, text generation | <a href="https://huggingface.co/Harsha001/Nanbeige4.3-33B-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FNanbeige4.3-33B-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Qwen3.8 27B Uncensored**](https://huggingface.co/Harsha001/Qwen3.8-27B-Uncensored-ROCmFPX) | orcarouter | 27B, image and text | <a href="https://huggingface.co/Harsha001/Qwen3.8-27B-Uncensored-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FQwen3.8-27B-Uncensored-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Arete-27B Uncensored**](https://huggingface.co/Harsha001/Arete-27B-Uncensored-ROCmFPX) | SargeDev | 27B, text generation | <a href="https://huggingface.co/Harsha001/Arete-27B-Uncensored-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FArete-27B-Uncensored-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**DaisyRose-27B**](https://huggingface.co/Harsha001/DaisyRose-27B-ROCmFPX) | RadixArk | 27B, image and text | <a href="https://huggingface.co/Harsha001/DaisyRose-27B-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FDaisyRose-27B-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Ternary Bonsai 2.5 27B Uncensored**](https://huggingface.co/Harsha001/Ternary-Bonsai-2.5-27B-Uncensored-ROCmFPX) | prism-ml | 27B ternary, <code>Q1_0_ROCMFPT</code> | <a href="https://huggingface.co/Harsha001/Ternary-Bonsai-2.5-27B-Uncensored-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FTernary-Bonsai-2.5-27B-Uncensored-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**humanizer 12B**](https://huggingface.co/Harsha001/humanizer-12b-ROCmFPX) | jialinyyzz | 12B, text generation | <a href="https://huggingface.co/Harsha001/humanizer-12b-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2Fhumanizer-12b-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**SenseNova-Loop-9B**](https://huggingface.co/Harsha001/SenseNova-Loop-9B-ROCmFPX) | sensenova | 9B looped model, image and text | <a href="https://huggingface.co/Harsha001/SenseNova-Loop-9B-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FSenseNova-Loop-9B-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**DeepSeek-V4-Pro Qwen3.5 9B heretic**](https://huggingface.co/Harsha001/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-ROCmFPX) | xtools-at | 9B, image and text | <a href="https://huggingface.co/Harsha001/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FDeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Alpie Seed 9B**](https://huggingface.co/Harsha001/Alpie-Seed-9B-ROCmFPX) | 169Pi | 9B, image and text | <a href="https://huggingface.co/Harsha001/Alpie-Seed-9B-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FAlpie-Seed-9B-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Sage-A1.5**](https://huggingface.co/Harsha001/Sage-A1.5-ROCmFPX) | InternScience | text generation | <a href="https://huggingface.co/Harsha001/Sage-A1.5-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FSage-A1.5-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Carbon-X**](https://huggingface.co/Harsha001/Carbon-X-ROCmFPX) | HuggingFaceBio | text generation | <a href="https://huggingface.co/Harsha001/Carbon-X-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FCarbon-X-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Nemotron 3.5 Bolt E4B**](https://huggingface.co/Harsha001/NVIDIA-Nemotron-3.5-Bolt-E4B-ROCmFPX) | NVIDIA | text generation | <a href="https://huggingface.co/Harsha001/NVIDIA-Nemotron-3.5-Bolt-E4B-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FNVIDIA-Nemotron-3.5-Bolt-E4B-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Kolibri-1.1**](https://huggingface.co/Harsha001/Kolibri-1.1-ROCmFPX) | Aleph Alpha | sparse attention, cross-layer KV | <a href="https://huggingface.co/Harsha001/Kolibri-1.1-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FKolibri-1.1-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Spark-X2.6 4B**](https://huggingface.co/Harsha001/Spark-X2.6-4B-ROCmFPX) | XHToken | 4B, text generation | <a href="https://huggingface.co/Harsha001/Spark-X2.6-4B-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FSpark-X2.6-4B-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Raon-1 nano**](https://huggingface.co/Harsha001/Raon-1-nano-ROCmFPX) | KRAFTON | hybrid attention, MTP head | <a href="https://huggingface.co/Harsha001/Raon-1-nano-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FRaon-1-nano-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**LFM3-VL 2B**](https://huggingface.co/Harsha001/LFM3-VL-2B-ROCmFPX) | LiquidAI | 2B, image and text | <a href="https://huggingface.co/Harsha001/LFM3-VL-2B-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FLFM3-VL-2B-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**LFM3 1.7B**](https://huggingface.co/Harsha001/LFM3-1.7B-ROCmFPX) | LiquidAI | 1.7B, text generation | <a href="https://huggingface.co/Harsha001/LFM3-1.7B-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FLFM3-1.7B-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**LFM3 1.7B Base**](https://huggingface.co/Harsha001/LFM3-1.7B-Base-ROCmFPX) | LiquidAI | 1.7B base model | <a href="https://huggingface.co/Harsha001/LFM3-1.7B-Base-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FLFM3-1.7B-Base-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**flint**](https://huggingface.co/Harsha001/flint-ROCmFPX) | RecursiveSI | tiny weights, needed the tensor-scale fix | <a href="https://huggingface.co/Harsha001/flint-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2Fflint-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**elephant-1**](https://huggingface.co/Harsha001/elephant-1-ROCmFPX) | baselabs | decision model, runs with <code>llama-decide</code> | <a href="https://huggingface.co/Harsha001/elephant-1-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2Felephant-1-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**mammoth-1**](https://huggingface.co/Harsha001/mammoth-1-ROCmFPX) | baselabs | decision model, runs with <code>llama-decide</code> | <a href="https://huggingface.co/Harsha001/mammoth-1-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2Fmammoth-1-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |

<sub>Collection: [ROCmFPX language models](https://huggingface.co/collections/Harsha001/rocmfpx-language-models-6add045c1e468c2d2afcf592)</sub>

### Speech models

| Model | Original by | What it is | Downloads |
|---|---|---|---|
| [**Maya2 native**](https://huggingface.co/Harsha001/Maya2-Native-GGUF) | maya-research | text to speech | <a href="https://huggingface.co/Harsha001/Maya2-Native-GGUF"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FMaya2-Native-GGUF&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Drama-3**](https://huggingface.co/Harsha001/Drama-3-ROCmFPX-GGUF) | Fish Audio | text to speech | <a href="https://huggingface.co/Harsha001/Drama-3-ROCmFPX-GGUF"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FDrama-3-ROCmFPX-GGUF&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Audio8-TTS Realistic 0.6B**](https://huggingface.co/Harsha001/Audio8-TTS-Realistic-0.6b-ROCmFPX) | Edge0 | text to speech | <a href="https://huggingface.co/Harsha001/Audio8-TTS-Realistic-0.6b-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FAudio8-TTS-Realistic-0.6b-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Haru-4B**](https://huggingface.co/Harsha001/Haru-4B-ROCmFPX) | higurashi-ai | full-duplex speech, runs with <code>llama-haru</code> | <a href="https://huggingface.co/Harsha001/Haru-4B-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FHaru-4B-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**Nemotron 3.5 Transcribe 30B-A4B**](https://huggingface.co/Harsha001/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-ROCmFPX) | NVIDIA | speech recognition | <a href="https://huggingface.co/Harsha001/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FNVIDIA-Nemotron-3.5-Transcribe-30B-A4B-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |

### Image and video models

GGUF files for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) with ROCmFPX weight types.

| Model | Original by | What it is | Downloads |
|---|---|---|---|
| [**Pika H3 Flex**](https://huggingface.co/Harsha001/Pika-H3-Flex-ROCmFPX) | Pika Labs | image to video (gated) | <a href="https://huggingface.co/Harsha001/Pika-H3-Flex-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FPika-H3-Flex-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**LTX-2.5 22B dev**](https://huggingface.co/Harsha001/LTX-2.5-22B-dev-uncensored-ROCmFPX) | Lightricks | text to video (gated) | <a href="https://huggingface.co/Harsha001/LTX-2.5-22B-dev-uncensored-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FLTX-2.5-22B-dev-uncensored-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**MiniMax H3 Image**](https://huggingface.co/Harsha001/MiniMax-H3-Image-ROCmFPX) | MiniMaxAI | text to image (gated) | <a href="https://huggingface.co/Harsha001/MiniMax-H3-Image-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FMiniMax-H3-Image-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**FLUX.3 image klein**](https://huggingface.co/Harsha001/flux-3-image-klein-ROCmFPX) | Black Forest Labs | text to image | <a href="https://huggingface.co/Harsha001/flux-3-image-klein-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2Fflux-3-image-klein-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |
| [**MOSS-Image SFT**](https://huggingface.co/Harsha001/MOSS-Image-SFT-ROCmFPX) | OpenMOSS | text to image | <a href="https://huggingface.co/Harsha001/MOSS-Image-SFT-ROCmFPX"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FMOSS-Image-SFT-ROCmFPX&query=%24.downloads&label=downloads%2F30d&color=ED1C24" alt="downloads in the last 30 days"></a> |

<sub>Collection: [ROCmFPX image and video models](https://huggingface.co/collections/Harsha001/rocmfpx-image-and-video-models-6add045e50d985721da52ac2)</sub>

### Speech on the AMD XDNA2 NPU

Not llama.cpp: these run on the Ryzen AI NPU through Windows ML. Same author, same hardware.

| Model | Original by | What it is | Downloads |
|---|---|---|---|
| [**Phonon-2**](https://huggingface.co/Harsha001/Phonon-2-XDNA2) | FermionResearch | speech recognition on the Ryzen AI NPU | <a href="https://huggingface.co/Harsha001/Phonon-2-XDNA2"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FPhonon-2-XDNA2&query=%24.downloads&label=downloads%2F30d&color=2b3137" alt="downloads in the last 30 days"></a> |
| [**Underdog Bark 3.0**](https://huggingface.co/Harsha001/Underdog-Bark-3.0-XDNA2) | ConwayResearch | speech recognition on the Ryzen AI NPU | <a href="https://huggingface.co/Harsha001/Underdog-Bark-3.0-XDNA2"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHarsha001%2FUnderdog-Bark-3.0-XDNA2&query=%24.downloads&label=downloads%2F30d&color=2b3137" alt="downloads in the last 30 days"></a> |

<sub>Collection: [Speech on the AMD XDNA2 NPU](https://huggingface.co/collections/Harsha001/speech-on-the-amd-xdna2-npu-6add045f8a9fcd3cba9182e4)</sub>

## Tests

On a Strix Halo PC (Windows, Radeon 8060S, Vulkan):

| Test | Result |
|---|---|
| `test-backend-ops -b Vulkan0 -p rocm` (every op with a ROCm type) | **2318 pass, 0 fail** |
| `test-backend-ops -b Vulkan0`: `MUL_MAT` / `MUL_MAT_ID` / `GET_ROWS` | 2577 / 1761 / 320 pass, 0 fail |
| `test-backend-ops -b Vulkan0`: `MUL_MAT_VEC_FUSION` / `MUL_MAT_ID_FUSION` / `MOE_UP_GATE_GLU` / `MUL_MAT_TENSOR_SCALE` | 2323 / 13 / 48 / 108 pass, 0 fail |
| `ctest` (tokenizers incl. Raon-1, chat parsers, quantize functions, GGUF, ...) | **47 of 47 pass** |
| `test-chat` (incl. the Inkling format) | all pass |
| `test-llama-archs` (builds, saves and reloads a small model of every architecture) | 173 of 173 pass; `hy_v4_flash` has no complete test model yet |
| Qwen3.6-35B-A3B greedy output, before and after merging all branches | identical (6 prompts) |

Earlier runs, still valid for the code they cover:

| Test | Result |
|---|---|
| `test-backend-ops -b ROCm0 -p rocm` (HIP SDK 7.2, gfx1151) | 1669 pass, 0 fail, the 7 original ROCm types |
| `llama-quantize` against charlie12345/ROCmFPX `fb08d7c` | byte-identical tensors for all 19 original ROCm recipes plus `Q4_K_M` and `Q8_0` |

## Branches

`main` holds everything above. The model branches (`hy4-flash`, `inkling2-mini`, `kolibri-1.1`, `raon-1-nano`, `rocmfpx-gptq-v2`, `rocmfpx-sub2`) are merged into it and stay on GitHub only so that older model cards keep working.

## Credits

- [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml by Georgi Gerganov and contributors.
- The original ROCmFPX and ROCmFP4 formats: [charlie12345 (Carlo Pasquale)](https://github.com/charlie12345/ROCmFPX), caf, ciru-ai and contributors.
- DeepSeek-V4.1-Flash support started from upstream pull request #30817; the uncensored DeepSeek weights are by Infatoshi, using the attention edit by pqhaz.
- The Inkling chat format parser is taken from upstream pull request #32381.

## License

MIT, like llama.cpp. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

---

<details>
<summary><b>The upstream llama.cpp README</b> (quick start, backends, documentation)</summary>

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

</details>
