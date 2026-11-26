#!/usr/bin/env python3
# Convert an Inkling 2 text model (thinkingmachines/Inkling-2-Small layout, model.llm.*) to GGUF.
#
# Streams one tensor at a time: tensor infos are computed from the safetensors headers first,
# then each tensor is loaded, converted and written, so the full model never sits in memory.
#
# Expert formats in the source:
#   float (BF16/F32)       -> written as is (F32 stays F32, BF16 stays BF16)
#   NVFP4 (U8 + E4M3 + amax / outer_scale_denominator)
#                          -> ggml NVFP4, bit-exact, plus a per-expert .scale = amax / denominator
#   MXFP8 (E4M3 + UE8M0 per 32)
#                          -> Q8_0 (default) or BF16 (--mxfp8 bf16)
# The w13 rows are interleaved (2i = gate_i, 2i+1 = up_i) and are split into gate and up.
# The vision tower and the MTP head are not converted.

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402

logger = logging.getLogger("inkling2")

QT = gguf.GGMLQuantizationType


# ---------------------------------------------------------------- source access

class Source:
    def __init__(self, model_dir: Path):
        self.dir = model_dir
        idx = model_dir / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
        else:
            wm = {}
            for f in sorted(model_dir.glob("*.safetensors")):
                with safe_open(f, "pt") as h:
                    for k in h.keys():
                        wm[k] = f.name
        self.weight_map = wm
        self._handles: dict[str, object] = {}
        self._meta: dict[str, tuple[str, list[int]]] = {}
        for fname in sorted(set(wm.values())):
            self._read_header(fname)

    def _read_header(self, fname: str):
        with open(self.dir / fname, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                self._meta[k] = (v["dtype"], v["shape"])

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def meta(self, name: str) -> tuple[str, list[int]]:
        return self._meta[name]

    def get(self, name: str) -> torch.Tensor:
        fname = self.weight_map[name]
        h: Any = self._handles.get(fname)
        if h is None:
            h = safe_open(self.dir / fname, "pt")
            self._handles[fname] = h
        return h.get_tensor(name)

    def names(self):
        return self.weight_map.keys()


# ---------------------------------------------------------------- formats

def nvfp4_pack(w: torch.Tensor, s: torch.Tensor) -> np.ndarray:
    """ModelOpt NVFP4 (byte = elem 2i | elem 2i+1 << 4, E4M3 scale per 16) -> ggml NVFP4 rows.
    ggml block of 64: 4 UE4M3 scales, then 4 x 8 bytes; in each 16-group byte j = elem j | elem j+8 << 4."""
    rows, nbytes = w.shape
    n_blocks = s.shape[1]
    assert nbytes * 2 == n_blocks * 16 and n_blocks % 4 == 0
    w = w.reshape(rows, n_blocks, 8)
    vals = torch.stack([w & 0x0F, w >> 4], dim=-1).reshape(rows, n_blocks, 16)
    qs = (vals[:, :, :8] | (vals[:, :, 8:] << 4)).to(torch.uint8).numpy()
    d = s.view(torch.uint8).numpy().reshape(rows, n_blocks) & 0x7F
    n_super = n_blocks // 4
    raw = np.concatenate([d.reshape(rows, n_super, 4), qs.reshape(rows, n_super, 32)], axis=-1)
    return raw.reshape(rows, n_super * 36)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def nvfp4_dequant(w: torch.Tensor, s: torch.Tensor, factor: float) -> torch.Tensor:
    """ModelOpt NVFP4 rows (even element in the low nibble, E4M3 scale per 16) -> F32, times the expert factor."""
    codes = w.to(torch.uint8).long()
    vals = torch.stack([E2M1[codes & 0xF], E2M1[codes >> 4]], dim=-1).reshape(w.shape[0], -1, 16)
    return (vals * s.float().unsqueeze(-1) * factor).reshape(w.shape[0], -1)


def mxfp8_dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """E4M3 [.., n] with UE8M0 per 32 -> F32."""
    x = w.float()
    e = s.view(torch.uint8).to(torch.int32) - 127
    scale = torch.pow(2.0, e.float())
    return (x.reshape(*x.shape[:-1], -1, 32) * scale.unsqueeze(-1)).reshape(x.shape)


def q8_0(x: torch.Tensor) -> np.ndarray:
    """F32 [rows, n] -> Q8_0 bytes [rows, n/32*34], as ggml quantize_row_q8_0_ref."""
    rows, n = x.shape
    b = x.reshape(rows, n // 32, 32)
    amax = b.abs().amax(dim=-1, keepdim=True)
    d = amax / 127.0
    idv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    q = b * idv
    q = (torch.sign(q) * torch.floor(q.abs() + 0.5)).to(torch.int8)
    dh = d.to(torch.float16).view(torch.uint8).reshape(rows, n // 32, 2)
    raw = torch.cat([dh, q.view(torch.uint8)], dim=-1)
    return raw.reshape(rows, n // 32 * 34).numpy()


def to_np(t: torch.Tensor, keep_bf16: bool = True) -> tuple[np.ndarray, QT | None]:
    if t.dtype == torch.bfloat16 and keep_bf16:
        return t.contiguous().view(torch.int16).numpy().view(np.uint16), QT.BF16
    if t.dtype in (torch.float16,):
        return t.contiguous().numpy(), None
    return t.float().contiguous().numpy(), None


# ---------------------------------------------------------------- config

def load_config(model_dir: Path) -> dict:
    c = json.loads((model_dir / "config.json").read_text())
    t = c.get("text_config", c)
    g = lambda *keys, default=None: next((t[k] for k in keys if k in t), default)  # noqa: E731

    n_layer: int = t["num_hidden_layers"]
    if "layer_types" in t:
        is_swa = [lt == "hybrid_sliding" for lt in t["layer_types"]]
    else:
        local = set(t["local_layer_ids"])
        is_swa = [i in local for i in range(n_layer)]

    for a, b in (("swa_head_dim", "head_dim"), ("swa_num_attention_heads", "num_attention_heads"),
                 ("swa_num_key_value_heads", "num_key_value_heads")):
        if a in t and t[a] != t[b]:
            raise ValueError(f"{a} != {b}: per-layer head shapes are not supported")

    cfg = dict(
        n_layer=n_layer,
        n_embd=g("hidden_size"),
        n_head=g("num_attention_heads"),
        n_head_kv=g("num_key_value_heads"),
        head_dim=g("head_dim"),
        d_rel=g("d_rel"),
        rel_extent=g("rel_extent"),
        n_swa=g("sliding_window_size"),
        is_swa=is_swa,
        eps=g("rms_norm_eps"),
        qk_clip=g("qk_l2_clip_threshold"),
        rel_clip=g("rel_l2_clip_threshold"),
        log_alpha=g("log_scaling_alpha"),
        log_floor=g("log_scaling_n_floor"),
        softcap=g("final_logit_softcapping"),
        gate_cap=g("mlp_gate_softcap"),
        up_cap=g("mlp_up_softcap"),
        route_scale=g("route_scale", "routed_scaling_factor"),
        n_expert=g("n_routed_experts", "num_local_experts"),
        n_expert_used=g("num_experts_per_tok"),
        n_ff_exp=g("moe_intermediate_size", "intermediate_size"),
        conv_k=g("sconv_kernel_size", "conv_kernel_size"),
        res_block=g("n_layers_per_attn_res_block"),
        vocab=g("unpadded_vocab_size", "vocab_size"),
        ctx=g("max_position_embeddings", "model_max_length", default=1048576),
        eos=c.get("eos_token_id", t.get("eos_token_id")),
        n_shared=g("n_shared_experts", default=0),
    )
    # Inkling 2 Mini: physical layers [loop_start, loop_end) run loop_count times, each visit gets a cache slot
    lc = t.get("loop_config") or {}
    cfg["loop_start"] = lc.get("core_start", 0)
    cfg["loop_end"] = lc.get("core_end", 0)
    cfg["loop_count"] = lc.get("num_passes", 1) if lc else 1
    if lc.get("is_varloop", False):
        raise ValueError("variable loops are not supported")
    span = cfg["loop_end"] - cfg["loop_start"]
    extra = span * (cfg["loop_count"] - 1)
    # slot of the first visit of a physical layer, and the physical layer of each slot
    cfg["first_slot"] = [i if i < cfg["loop_end"] else i + extra for i in range(n_layer)]
    slot_phys = list(range(cfg["loop_end"])) + [cfg["loop_start"] + j % span for j in range(extra)] \
        + list(range(cfg["loop_end"], n_layer))
    cfg["n_slot"] = len(slot_phys)
    cfg["slot_is_swa"] = [is_swa[p] for p in slot_phys]
    if cfg["n_shared"]:
        raise ValueError("shared experts are not supported")
    if g("scoring_func", default="sigmoid") != "sigmoid":
        raise ValueError("only the sigmoid router is supported")
    return cfg


# ---------------------------------------------------------------- tensor plan

class Item:
    def __init__(self, name, shape, qtype, produce):
        self.name = name          # gguf name
        self.shape = shape        # numpy order (outermost first)
        self.qtype = qtype        # QT
        self.produce = produce    # () -> np.ndarray (raw bytes for quantized, values otherwise)

    def nbytes(self) -> int:
        bs, ts = gguf.GGML_QUANT_SIZES[self.qtype]
        n = int(np.prod(self.shape))
        return n // bs * ts


class Plan(list):
    lora_used: set


def src_qtype(dtype: str) -> QT:
    return {"F32": QT.F32, "BF16": QT.BF16, "F16": QT.F16}[dtype]


def load_lora(path: Path, scale: float) -> tuple[dict, float]:
    """llama.cpp LoRA adapter GGUF -> {base tensor name: (a, b)} in numpy order, and the merge scale.
    a: [.., r, n_in] (token_embd: [V, r]), b: [.., n_out, r]; merged W = W + scale * b @ a (token_embd: a @ b.T)."""
    r = gguf.GGUFReader(path)
    assert bytes(r.fields["general.type"].parts[-1]).decode() == "adapter"
    alpha = float(r.fields["adapter.lora_alpha"].parts[-1][0])
    t = {x.name: np.array(x.data, dtype=np.float32) for x in r.tensors}
    pairs = {}
    for n in t:
        if n.endswith(".lora_a"):
            base = n[:-len(".lora_a")]
            pairs[base] = (t[n], t[base + ".lora_b"])
    a0, _ = next(iter(pairs.values()))
    rank = a0.shape[-1] if next(iter(pairs)).startswith("token_embd") else a0.shape[-2]
    eff = scale * (alpha / rank if alpha else 1.0)
    logger.info(f"LoRA {path.name}: {len(pairs)} tensors, alpha {alpha}, rank {rank}, merge scale {eff}")
    return pairs, eff


def build_plan(src: Source, cfg: dict, mxfp8_mode: str, nvfp4_mode: str = "native", all_f32: bool = False,
               lora: dict | None = None, lora_scale: float = 1.0) -> Plan:
    P = "model.llm."
    plan: Plan = Plan()
    V = cfg["vocab"]
    K = cfg["conv_k"]
    # LoRA names use physical layers (blk.i); our tensors live under the slot of the first visit
    lora_map = {}
    for n, ab in (lora or {}).items():
        if n.startswith("blk."):
            i, rest = n[4:].split(".", 1)
            n = f"blk.{cfg['first_slot'][int(i)]}.{rest}"
        lora_map[n] = ab
    lora_used = set()

    def lora_delta(gname: str) -> np.ndarray | None:
        if gname not in lora_map:
            return None
        lora_used.add(gname)
        a, b = lora_map[gname]
        if gname.startswith("token_embd"):
            return lora_scale * (a @ b.T)
        return lora_scale * np.matmul(b, a)

    def plain(gname, sname, f32=False, transform=None, shape=None):
        dt, sh = src.meta(sname)
        has_lora = gname in lora_map
        qt = QT.F32 if f32 or all_f32 or has_lora else src_qtype(dt)
        shp = shape if shape is not None else sh

        def produce():
            t = src.get(sname)
            if transform is not None:
                t = transform(t)
            if qt == QT.F32:
                x = t.float().contiguous().numpy()
                d = lora_delta(gname)
                if d is not None:
                    if gname.startswith("token_embd") and d.shape[0] > x.shape[0]:
                        # adapters made on the padded upstream embedding: token ids match, padding rows are unused
                        d = d[:x.shape[0]]
                    assert d.shape == x.shape, (gname, d.shape, x.shape)
                    x = x + d.astype(np.float32)
                return x
            return to_np(t)[0]
        plan.append(Item(gname, list(shp), qt, produce))

    plain("token_embd.weight", P + "embed.weight", transform=lambda t: t[:V], shape=[V, cfg["n_embd"]])
    plain("token_embd_norm.weight", P + "embed_norm.weight", f32=True)
    plain("output_norm.weight", P + "norm.weight", f32=True)
    plain("output_res_q.weight", P + "attn_res.output_query", f32=True)

    cache: dict = {}

    def expert_parts(il: int, which: str, force_f32: bool = False):
        # returns a callable per part ('gate', 'up', 'down') and its (shape, qtype)
        base = f"{P}layers.{il}.mlp.experts."
        sname = base + ("w13_weight" if which in ("gate", "up") else "w2_weight")
        dt, sh = src.meta(sname)
        E = sh[0]
        sel = {"gate": slice(0, None, 2), "up": slice(1, None, 2), "down": slice(None)}[which]

        def load(name):
            key = (il, name)
            if key not in cache:
                cache.clear()
                cache[key] = src.get(name)
            return cache[key]

        if dt in ("F32", "BF16", "F16"):
            rows = sh[1] // 2 if which != "down" else sh[1]
            shape = [E, rows, sh[2]]
            qt = QT.F32 if force_f32 else src_qtype(dt)

            def produce():
                t = load(sname)[:, sel]
                return t.float().contiguous().numpy() if qt == QT.F32 else to_np(t)[0]
            return shape, qt, produce, None

        scale_name = sname + "_scale"
        amax_name = sname + "_amax"
        if src.has(amax_name) and (nvfp4_mode == "f32" or force_f32):
            # NVFP4 dequantized (test models narrower than one ggml NVFP4 block)
            rows = sh[1] // 2 if which != "down" else sh[1]
            ncols = sh[2] * 2
            shape = [E, rows, ncols]

            def produce():
                w = load(sname)[:, sel]
                s = src.get(scale_name)[:, sel]
                f = (src.get(amax_name).float().reshape(-1) / src.get(sname + "_outer_scale_denominator").float().reshape(-1))
                out = np.empty((E, rows, ncols), dtype=np.float32)
                for e in range(E):
                    out[e] = nvfp4_dequant(w[e].contiguous(), s[e].contiguous(), float(f[e])).numpy()
                return out
            return shape, QT.F32, produce, None

        if src.has(amax_name):
            # NVFP4
            rows = sh[1] // 2 if which != "down" else sh[1]
            ncols = sh[2] * 2
            shape = [E, rows, ncols]

            def produce():
                w = load(sname)[:, sel]
                s = src.get(scale_name)[:, sel]
                out = np.empty((E, rows, ncols // 64 * 36), dtype=np.uint8)
                for e in range(E):
                    out[e] = nvfp4_pack(w[e].contiguous(), s[e].contiguous())
                return out

            def produce_scale():
                amax = src.get(amax_name).float().reshape(-1)
                den = src.get(sname + "_outer_scale_denominator").float().reshape(-1)
                assert amax.numel() == E
                return (amax / den).contiguous().numpy().astype(np.float32)
            return shape, QT.NVFP4, produce, produce_scale

        # MXFP8
        rows = sh[1] // 2 if which != "down" else sh[1]
        ncols = sh[2]
        shape = [E, rows, ncols]
        if mxfp8_mode == "q8_0" and not force_f32:
            def produce():
                w = load(sname)[:, sel]
                s = src.get(scale_name)[:, sel]
                out = np.empty((E, rows, ncols // 32 * 34), dtype=np.uint8)
                for e in range(E):
                    out[e] = q8_0(mxfp8_dequant(w[e], s[e]))
                return out
            return shape, QT.Q8_0, produce, None
        if mxfp8_mode == "f32" or all_f32 or force_f32:
            def produce():
                w = load(sname)[:, sel]
                s = src.get(scale_name)[:, sel]
                return np.stack([mxfp8_dequant(w[e], s[e]).numpy() for e in range(E)]).astype(np.float32)
            return shape, QT.F32, produce, None

        def produce():
            w = load(sname)[:, sel]
            s = src.get(scale_name)[:, sel]
            out = np.empty((E, rows, ncols), dtype=np.uint16)
            for e in range(E):
                out[e] = mxfp8_dequant(w[e], s[e]).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            return out
        return shape, QT.BF16, produce, None

    looped = cfg["loop_count"] > 1
    if looped:
        # [q, mlp] and [k, v] per loop layer -> [n_loop, 4, n_embd] in q, k, v, mlp order
        n_loop = cfg["loop_end"] - cfg["loop_start"]

        def produce_msg_q():
            q = src.get(P + "middle_cycle_cross_pass_queries").float().reshape(n_loop, 2, -1)
            kv = src.get(P + "middle_cycle_cross_pass_kv_queries").float().reshape(n_loop, 2, -1)
            return torch.cat([q[:, :1], kv, q[:, 1:]], dim=1).contiguous().numpy()
        plan.append(Item("loop_msg_q.weight", [n_loop, 4, cfg["n_embd"]], QT.F32, produce_msg_q))

    for il in range(cfg["n_layer"]):
        L = f"{P}layers.{il}."
        B = f"blk.{cfg['first_slot'][il]}."
        swa = cfg["is_swa"][il]

        if looped and cfg["loop_start"] <= il < cfg["loop_end"]:
            # message gates stacked in q (controllers.attn), k, v, mlp order
            ctl = [f"{L}controllers.{c}." for c in ("attn", "k", "v", "mlp")]
            for gname, sname, f32 in (("loop_gate_down", "down_weight", False),
                                      ("loop_gate_forget", "forget_weight", True),
                                      ("loop_gate_msg", "gain_weight", True)):
                dt, sh = src.meta(ctl[0] + sname)
                qt = QT.F32 if f32 or all_f32 else src_qtype(dt)

                def produce(names=[c + sname for c in ctl], qt=qt):
                    t = torch.stack([src.get(n) for n in names], dim=0)
                    return t.float().contiguous().numpy() if qt == QT.F32 else to_np(t)[0]
                plan.append(Item(B + gname + ".weight", [4, *sh], qt, produce))

        if il > 0:
            plain(B + "attn_res_q.weight", f"{P}attn_res.attn_queries.{il - 1}", f32=True)
        plain(B + "ffn_res_q.weight", f"{P}attn_res.mlp_queries.{il}", f32=True)

        if swa:
            plain(B + "attn_norm.weight", L + "attn_norm.weight", f32=True)
        else:
            for x in "qkv":
                plain(B + f"attn_{x}_in_norm.weight", L + f"{x}_input_norm.weight", f32=True)

        plain(B + "attn_q.weight", L + "attn.wq_du.weight")
        plain(B + "attn_k.weight", L + "attn.wk_dv.weight")
        plain(B + "attn_v.weight", L + "attn.wv_dv.weight")
        plain(B + "attn_output.weight", L + "attn.wo_ud.weight")
        plain(B + "attn_gate.weight", L + "attn.gate_attn.weight")
        plain(B + "attn_sink_proj.weight", L + "attn.attn_sink_proj.weight")
        plain(B + "attn_rel.weight", L + "attn.wr_du.weight")

        _, rp = src.meta(L + "attn.rel_logits_proj.proj")
        plain(B + "attn_rel_proj.weight", L + "attn.rel_logits_proj.proj", f32=True,
              transform=lambda t: t.t(), shape=[rp[1], rp[0]])
        plain(B + "attn_q_clip.weight", L + "attn.q_l2_clip.scale", f32=True)
        plain(B + "attn_rel_clip.weight", L + "attn.rel_l2_clip.scale", f32=True)

        for x in "kv":
            if swa:
                _, cs = src.meta(L + f"attn.{x}_sconv.weight")
                plain(B + f"attn_{x}_conv.weight", L + f"attn.{x}_sconv.weight", f32=True,
                      transform=lambda t: t[:, 0, :], shape=[cs[0], cs[2]])
            else:
                # static_w[lag] weights the token lag steps back; ssm_conv index K-1 is the current token
                _, cs = src.meta(L + f"attn.{x}_sconv.static_w")
                assert cs[0] == K
                plain(B + f"attn_{x}_conv.weight", L + f"attn.{x}_sconv.static_w", f32=True,
                      transform=lambda t: t.flip(0).t(), shape=[cs[1], cs[0]])
                plain(B + f"attn_{x}_conv_dyn.weight", L + f"attn.{x}_sconv.weight_proj", f32=True)

        _, cs = src.meta(L + "attn_sconv.weight")
        plain(B + "attn_post_conv.weight", L + "attn_sconv.weight", f32=True,
              transform=lambda t: t[:, 0, :], shape=[cs[0], cs[2]])
        plain(B + "ffn_post_conv.weight", L + "mlp_sconv.weight", f32=True,
              transform=lambda t: t[:, 0, :], shape=[cs[0], cs[2]])

        plain(B + "ffn_norm.weight", L + "mlp_norm.weight", f32=True)
        plain(B + "ffn_gate_inp.weight", L + "mlp.gate.weight", f32=True)
        plain(B + "exp_probs_b.bias", L + "mlp.gate.bias", f32=True)

        for part, gname in (("gate", "ffn_gate_exps"), ("up", "ffn_up_exps"), ("down", "ffn_down_exps")):
            lname = B + gname + ".weight"
            shape, qt, produce, produce_scale = expert_parts(il, part, force_f32=lname in lora_map)
            if lname in lora_map:
                assert qt == QT.F32 and produce_scale is None

                def produce(base=produce, lname=lname):
                    x = base()
                    d = lora_delta(lname)
                    assert d is not None
                    assert d.shape == x.shape, (lname, d.shape, x.shape)
                    x += d.astype(np.float32)
                    return x
            plan.append(Item(lname, shape, qt, produce))
            if produce_scale is not None:
                plan.append(Item(B + gname + ".scale", [shape[0]], QT.F32, produce_scale))

    plain("output.weight", P + "unembed.weight", transform=lambda t: t[:V], shape=[V, cfg["n_embd"]])
    missing = set(lora_map) - {it.name for it in plan}
    assert not missing, f"LoRA tensors with no target: {sorted(missing)[:5]}"
    plan.lora_used = lora_used  # filled while producing; checked after writing
    return plan


# ---------------------------------------------------------------- vocab

def add_vocab(w: gguf.GGUFWriter, tok_dir: Path, n_vocab: int):
    tj = json.loads((tok_dir / "tokenizer.json").read_text(encoding="utf-8"))
    model = tj["model"]
    assert model["type"] == "BPE"
    tokens = [None] * n_vocab
    types = [gguf.TokenType.NORMAL] * n_vocab
    for t, i in model["vocab"].items():
        if i < n_vocab:
            tokens[i] = t
    for a in tj.get("added_tokens", []):
        i = a["id"]
        if i < n_vocab:
            tokens[i] = a["content"]
            types[i] = gguf.TokenType.CONTROL if a.get("special") else gguf.TokenType.USER_DEFINED
    for i in range(n_vocab):
        if tokens[i] is None:
            tokens[i] = f"[PAD{i}]"
            types[i] = gguf.TokenType.UNUSED
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]

    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-4o")
    w.add_token_list([str(t) for t in tokens])  # every slot is filled above
    w.add_token_types(types)
    w.add_token_merges(merges)

    ids = {t: i for i, t in enumerate(tokens)}
    if "<|endoftext|>" in ids:
        w.add_pad_token_id(ids["<|endoftext|>"])
    if "<|begin_of_text|>" in ids:
        w.add_bos_token_id(ids["<|begin_of_text|>"])
    w.add_add_bos_token(False)

    ct = tok_dir / "chat_template.jinja"
    if ct.exists():
        w.add_chat_template(ct.read_text(encoding="utf-8"))
    return ids


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("outfile", type=Path)
    ap.add_argument("--tokenizer-dir", type=Path, default=None)
    ap.add_argument("--mxfp8", choices=["q8_0", "bf16", "f32"], default="q8_0")
    ap.add_argument("--nvfp4", choices=["native", "f32"], default="native")
    ap.add_argument("--all-f32", action="store_true", help="store every plain tensor as F32 (exact reference checks)")
    ap.add_argument("--name", default="Inkling-2-Small")
    ap.add_argument("--lora", type=Path, default=None, help="llama.cpp LoRA adapter GGUF to merge (physical layer names)")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = load_config(args.model_dir)
    src = Source(args.model_dir)
    lora, lscale = load_lora(args.lora, args.lora_scale) if args.lora else (None, 1.0)
    plan = build_plan(src, cfg, args.mxfp8, args.nvfp4, args.all_f32, lora, lscale)
    total = sum(it.nbytes() for it in plan)
    logger.info(f"{len(plan)} tensors, {total / 2**30:.2f} GiB")
    if args.dry_run:
        for it in plan[:60]:
            logger.info(f"  {it.name} {it.shape} {it.qtype.name}")
        return

    arch = "inkling2"
    w = gguf.GGUFWriter(args.outfile, arch)
    w.add_name(args.name)
    w.add_block_count(cfg["n_slot"])
    w.add_context_length(cfg["ctx"])
    w.add_embedding_length(cfg["n_embd"])
    w.add_feed_forward_length(cfg["n_ff_exp"])
    w.add_head_count(cfg["n_head"])
    w.add_head_count_kv(cfg["n_head_kv"])
    w.add_key_length(cfg["head_dim"])
    w.add_value_length(cfg["head_dim"])
    w.add_layer_norm_rms_eps(cfg["eps"])
    w.add_sliding_window(cfg["n_swa"])
    w.add_sliding_window_pattern(cfg["slot_is_swa"])
    if cfg["loop_count"] > 1:
        w.add_uint32(f"{arch}.loop.start", cfg["loop_start"])
        w.add_uint32(f"{arch}.loop.end", cfg["loop_end"])
        w.add_uint32(f"{arch}.loop.count", cfg["loop_count"])
    w.add_expert_count(cfg["n_expert"])
    w.add_expert_used_count(cfg["n_expert_used"])
    w.add_expert_feed_forward_length(cfg["n_ff_exp"])
    w.add_expert_weights_scale(float(cfg["route_scale"]))
    w.add_expert_weights_norm(True)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_final_logit_softcapping(float(cfg["softcap"]))
    w.add_float32(f"{arch}.activation.situ_beta", float(cfg["gate_cap"]))
    w.add_float32(f"{arch}.activation.situ_linear_beta", float(cfg["up_cap"]))
    w.add_uint32(f"{arch}.attention.rel_extent", cfg["rel_extent"])
    w.add_uint32(f"{arch}.attention.rel_dim", cfg["d_rel"])
    w.add_float32(f"{arch}.attention.qk_l2_clip", float(cfg["qk_clip"]))
    w.add_float32(f"{arch}.attention.rel_l2_clip", float(cfg["rel_clip"]))
    w.add_float32(f"{arch}.attention.log_scaling_alpha", float(cfg["log_alpha"]))
    w.add_float32(f"{arch}.attention.log_scaling_n_floor", float(cfg["log_floor"]))
    w.add_uint32(f"{arch}.attn_res.block_layers", cfg["res_block"])
    w.add_uint32(f"{arch}.conv_kernel", cfg["conv_k"])
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)

    tok_dir = args.tokenizer_dir or args.model_dir
    add_vocab(w, tok_dir, cfg["vocab"])
    if cfg["eos"] is not None:
        eos = cfg["eos"] if isinstance(cfg["eos"], int) else cfg["eos"][0]
        w.add_eos_token_id(eos)

    for it in plan:
        np_dtype = np.float32 if it.qtype == QT.F32 else (np.float16 if it.qtype == QT.F16 else np.uint8)
        if it.qtype in (QT.F32, QT.F16):
            w.add_tensor_info(it.name, it.shape, np.dtype(np_dtype), it.nbytes())
        else:
            # byte shape, the writer turns it back into the element shape
            bs, ts = gguf.GGML_QUANT_SIZES[it.qtype]
            byte_shape = [*it.shape[:-1], it.shape[-1] // bs * ts]
            w.add_tensor_info(it.name, byte_shape, np.dtype(np.uint8), it.nbytes(), raw_dtype=it.qtype)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    done = 0
    for i, it in enumerate(plan):
        arr = it.produce()
        if it.qtype not in (QT.F32, QT.F16):
            arr = np.ascontiguousarray(arr).view(np.uint8)
        else:
            arr = np.ascontiguousarray(arr)
        if arr.nbytes != it.nbytes():
            raise RuntimeError(f"{it.name}: {arr.nbytes} bytes, expected {it.nbytes()}")
        w.write_tensor_data(arr)
        done += arr.nbytes
        if i % 20 == 0 or it.nbytes() > 2**28:
            logger.info(f"[{i + 1}/{len(plan)}] {it.name} {it.qtype.name} {it.shape}  {done / 2**30:.1f} GiB")
    w.close()
    if lora:
        assert len(plan.lora_used) == len(lora), f"LoRA merged into {len(plan.lora_used)} of {len(lora)} tensors"
        logger.info(f"LoRA merged into {len(plan.lora_used)} tensors")
    logger.info(f"wrote {args.outfile} ({done / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
