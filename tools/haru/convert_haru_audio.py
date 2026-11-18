#!/usr/bin/env python3
"""Audio parts of Haru-4B (higurashi-ai/Haru-4B) as a GGUF for llama-haru.

    convert_haru_audio.py CHECKPOINT_DIR OUT.gguf [--type f32|f16|q8_0]

The text backbone ("lm.*") is a plain qwen35 model: convert it with convert_hf_to_gguf.py. This file holds
everything else under its PyTorch name: the two audio projectors (model_audio_proj.*, user_audio_proj.*), the
MeanFlow head (head.*), the VibeVoice codec (codec.encoder.*, codec.decoder.*), the latent scale and bias, and
float32 copies of the PAD / EOP embedding rows (the runtime computes those two logits in float32, as the
reference does, because the stay-silent / start-speaking decision is often closer than half precision).

--type sets the 2-D linear weights of the projectors, head and codec MLPs (q8_0 needs rows divisible by 32).
Convolution kernels, norms, biases and scales stay float32.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402
from safetensors import safe_open  # noqa: E402

ARCH = "haru-audio"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--type", choices=("f32", "f16", "q8_0"), default="f16")
    a = ap.parse_args()

    cfg = json.loads((a.ckpt / "config.json").read_text(encoding="utf-8"))
    codec, head = cfg["codec"], cfg["head"]
    w = gguf.GGUFWriter(str(a.out), ARCH)
    w.add_name("Haru-4B audio")
    kv = w.add_uint32
    kv(f"{ARCH}.sample_rate", cfg["sample_rate"])
    kv(f"{ARCH}.frame_samples", cfg["frame_samples"])
    kv(f"{ARCH}.latent_dim", cfg["latent_dim"])
    kv(f"{ARCH}.pad_token_id", cfg["pad_token_id"])
    kv(f"{ARCH}.eop_token_id", cfg["eop_token_id"])
    kv(f"{ARCH}.eos_token_id", cfg["lm"]["eos_token_id"])
    kv(f"{ARCH}.hidden_size", cfg["lm"]["hidden_size"])
    kv(f"{ARCH}.codec.base_channels", codec["base_channels"])
    w.add_array(f"{ARCH}.codec.ratios", [int(x) for x in codec["ratios"]])
    w.add_array(f"{ARCH}.codec.encoder_depths", [int(x) for x in codec["encoder_depths"]])
    w.add_array(f"{ARCH}.codec.decoder_depths", [int(x) for x in codec["decoder_depths"]])
    w.add_float32(f"{ARCH}.codec.norm_eps", codec["norm_eps"])
    w.add_float32(f"{ARCH}.codec.latent_std", codec["latent_std"])
    kv(f"{ARCH}.head.num_layers", head["num_layers"])
    kv(f"{ARCH}.head.ffn_size", int(head["hidden_size"] * head["ffn_ratio"]))
    w.add_float32(f"{ARCH}.head.norm_eps", head["norm_eps"])
    w.add_float32(f"{ARCH}.head.interval_embed", head["interval_embed"])
    kv(f"{ARCH}.head.num_steps", head["num_steps"])
    w.add_float32(f"{ARCH}.head.guidance_scale", head["guidance_scale"])
    w.add_array(f"{ARCH}.head.guidance_range", [float(x) for x in head["guidance_range"]])

    big = {"f32": gguf.GGMLQuantizationType.F32, "f16": gguf.GGMLQuantizationType.F16,
           "q8_0": gguf.GGMLQuantizationType.Q8_0}[a.type]
    n = 0
    with safe_open(str(a.ckpt / "model.safetensors"), "pt") as f:
        for name in f.keys():
            if name.startswith("lm."):
                if name == "lm.embed_tokens.weight":
                    emb = f.get_tensor(name)
                    rows = emb[[cfg["pad_token_id"], cfg["eop_token_id"]]].float().numpy()
                    w.add_tensor("stream_embd", np.ascontiguousarray(rows))  # [2, hidden]: PAD, EOP
                    n += 1
                continue
            t = f.get_tensor(name).float().numpy()
            if t.ndim == 0:
                t = t.reshape(1)
            linear = t.ndim == 2 and name.endswith(".weight")
            if t.ndim == 3:  # convolutions, in the layout of the llama-haru graphs (kept float32)
                if ".stages." in name:  # depthwise [C, 1, K] -> [K, C]: one contiguous row of channel weights per tap
                    t = t.reshape(t.shape[0], t.shape[2]).T
                elif ".up." in name:  # transposed [I, O, 2s] -> [2s * O, I]: one matmul gives every tap
                    t = t.transpose(2, 1, 0).reshape(-1, t.shape[0])
                else:  # dense [O, I, K] -> [O, K * I], tap-major, to multiply unfolded windows
                    t = t.transpose(0, 2, 1).reshape(t.shape[0], -1)
                t = np.ascontiguousarray(t)
                linear = False
            qt = big if linear else gguf.GGMLQuantizationType.F32
            if qt == gguf.GGMLQuantizationType.Q8_0 and t.shape[-1] % 32:
                qt = gguf.GGMLQuantizationType.F16
            if qt == gguf.GGMLQuantizationType.F16:
                t = t.astype(np.float16)
            elif qt == gguf.GGMLQuantizationType.Q8_0:
                t = gguf.quants.quantize(t, qt)
            assert len(name) < 64, name
            w.add_tensor(name, t, raw_dtype=qt if qt == gguf.GGMLQuantizationType.Q8_0 else None)
            n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out} ({n} tensors, linear weights {a.type})")


if __name__ == "__main__":
    main()
