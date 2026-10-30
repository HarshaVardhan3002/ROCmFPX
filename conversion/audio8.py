from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, TextModel, gguf, logger

# Audio8 (Edge0/Audio8-TTS-Realistic-0.6b, arktts) is a dual AR TTS model:
# a Qwen2-like slow AR picks the semantic token, a 4 layer fast AR gives codebooks 1..9,
# and the arktts codec decoder turns the 10-code frames into 44.1 kHz PCM.
#
# Tricks being used to support this model via existing llama.cpp code paths:
# - the slow AR is a plain llama graph: qkv bias, rope on interleaved pairs (llama.cpp NORM rope,
#   so q/k are not permuted), tied output
# - the input of a semantic position is embeddings[semantic] + sum_i codebook_embeddings[code_i + i * 4096].
#   code 0 is semantic - semantic_begin, so the semantic rows are added to the first 4096 table rows here
#   and the frame feedback is one table sum
# - the codec RVQ out_proj (weight norm + bias) is folded into each codebook, decode is get_rows + add
# - the codec post transformer uses rope on interleaved pairs, its q/k rows are permuted here
#   so the qwen3-tts code2wav layer (NEOX rope) gives the same attention
#
# pipeline stage mapping:
#   slow AR            --> mapped to normal libllama text model (autoregressive)
#   fast AR            --> MTMD_GEN_PROCESS_TYPE_GEN_CODE (+ the frame feedback embedding)
#   codec decoder      --> MTMD_GEN_PROCESS_TYPE_GEN_WAV


def _split_qkv(w: Tensor, n_head: int, n_head_kv: int, head_dim: int) -> tuple[Tensor, Tensor, Tensor]:
    nq, nkv = n_head * head_dim, n_head_kv * head_dim
    return w[:nq], w[nq:nq + nkv], w[nq + nkv:nq + 2 * nkv]


def _pairs_to_halves(w: Tensor, head_dim: int) -> Tensor:
    # rows of each head go from (x0, x1, x2, x3, ..) to (x0, x2, .., x1, x3, ..)
    # so NEOX rope over (i, i + d/2) rotates the original pairs (2i, 2i + 1)
    shape = w.shape
    w = w.reshape(-1, head_dim // 2, 2, *shape[1:])
    return w.transpose(1, 2).reshape(shape)


def _weight_norm(g: Tensor, v: Tensor) -> Tensor:
    # torch weight_norm with dim=0: w = g * v / ||v||, the norm over all dims but the first
    v = v.float()
    dims = tuple(range(1, v.ndim))
    return g.float() * v / v.norm(dim=dims, keepdim=True)


@ModelBase.register("ArkttsModel")
# [TAG_HF_EXAMPLE_MISSING] needs trust_remote_code, the codec lives in codec.pth
class Audio8Model(TextModel):
    model_arch = gguf.MODEL_ARCH.LLAMA

    def get_vocab_base_pre(self, tokenizer) -> str:
        del tokenizer
        # same pre-tokenizer regex as qwen2, the hash differs because of the 4096 semantic tokens
        return "qwen2"

    def set_vocab(self):
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

        self.gguf_writer.add_add_bos_token(False)
        # this is not a chat model
        self.gguf_writer.add_chat_template(None)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_dimension_count(self.hparams["head_dim"])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if name.startswith(("fast_", "codebook_embeddings")):
            return None  # the fast AR goes to the mmproj
        return item

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        hp = self.hparams

        if name == "embeddings.weight":
            yield (self.format_tensor_name(T.TOKEN_EMBD), data_torch)
            return
        if name == "norm.weight":
            yield (self.format_tensor_name(T.OUTPUT_NORM), data_torch)
            return

        assert bid is not None and name.startswith(f"layers.{bid}."), name
        key = name[len(f"layers.{bid}."):]
        if key.startswith("attention.wqkv."):
            suffix = "." + key.rsplit(".", 1)[1]
            q, k, v = _split_qkv(data_torch, hp["n_head"], hp["n_local_heads"], hp["head_dim"])
            yield (self.format_tensor_name(T.ATTN_Q, bid, suffix=suffix), q)
            yield (self.format_tensor_name(T.ATTN_K, bid, suffix=suffix), k)
            yield (self.format_tensor_name(T.ATTN_V, bid, suffix=suffix), v)
            return

        simple = {
            "attention.wo.weight":   T.ATTN_OUT,
            "attention_norm.weight": T.ATTN_NORM,
            "ffn_norm.weight":       T.FFN_NORM,
            "feed_forward.w1.weight": T.FFN_GATE,
            "feed_forward.w3.weight": T.FFN_UP,
            "feed_forward.w2.weight": T.FFN_DOWN,
        }
        yield (self.format_tensor_name(simple[key], bid), data_torch)


@ModelBase.register("ArkttsModel")
# [TAG_HF_EXAMPLE_MISSING] needs trust_remote_code, the codec lives in codec.pth
class Audio8MmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = False

    _FAST_MAP = {
        "attention.wo.weight":    gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_OUT,
        "attention_norm.weight":  gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_NORM,
        "ffn_norm.weight":        gguf.MODEL_TENSOR.A_GEN_CODE_FFN_NORM,
        "feed_forward.w1.weight": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_GATE,
        "feed_forward.w3.weight": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_UP,
        "feed_forward.w2.weight": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_DOWN,
    }
    _TFM_MAP = {
        "attention.wo.weight":            gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_OUT,
        "attention_norm.weight":          gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_NORM,
        "ffn_norm.weight":                gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_NORM,
        "feed_forward.w1.weight":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_GATE,
        "feed_forward.w3.weight":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_UP,
        "feed_forward.w2.weight":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_DOWN,
        "attention_layer_scale.gamma":    gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_SCALE,
        "ffn_layer_scale.gamma":          gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_SCALE,
    }

    def get_audio_config(self) -> dict[str, Any] | None:
        # the fast AR is the gen-audio part, the codec is hardcoded in clip.cpp
        g = self.global_config
        return {
            "n_layer":           g["n_fast_layer"],
            "hidden_size":       g["fast_dim"],
            "intermediate_size": g["fast_intermediate_size"],
            "num_heads":         g["fast_n_head"],
            "num_kv_heads":      g["fast_n_local_heads"],
            "head_dim":          g["fast_head_dim"],
            "norm_eps":          g["norm_eps"],
        }

    def set_gguf_parameters(self):
        g = self.global_config
        fast = self.hparams_audio
        assert fast is not None
        assert g["fast_dim"] == g["dim"] and not g["fast_attention_qkv_bias"] and not g["fast_attention_qk_norm"]
        assert g["norm_fastlayer_input"] and g["num_codebooks"] == 10 and g["codebook_size"] == 4096

        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.AUDIO8_GEN)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(fast["hidden_size"])
        self.gguf_writer.add_gen_audio_feed_forward_length(fast["intermediate_size"])
        self.gguf_writer.add_gen_audio_block_count(fast["n_layer"])
        self.gguf_writer.add_gen_audio_head_count(fast["num_heads"])
        self.gguf_writer.add_gen_audio_head_count_kv(fast["num_kv_heads"])
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(fast["norm_eps"])
        self.gguf_writer.add_gen_audio_model_variant("audio8")

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del name, bid, n_dims
        # conv1d/conv1d_dw kernels must be F16, ggml_conv_1d(_dw) has no BF16 path
        if new_name in ("a.gen.wav.dac.entry.weight", "a.gen.wav.dac.post_conv.weight") or (
            (".up.blk." in new_name and new_name.endswith(".dwconv.weight"))
            or (".dac.blk." in new_name and new_name.endswith((".conv1.weight", ".conv2.weight")))
        ):
            return gguf.GGMLQuantizationType.F16
        # ConvTranspose1d kernels: only F16/F32 are implemented, no BF16
        if new_name.endswith(".conv.weight") and (".up.blk." in new_name or ".dac.blk." in new_name):
            return gguf.GGMLQuantizationType.F32
        # get_rows sources: the folded codebooks stay exact, the code tables go F16
        if new_name.startswith("a.gen.wav.quant."):
            return gguf.GGMLQuantizationType.F32
        if new_name.startswith(("a.gen.audio_embd", "a.gen.code.embd")):
            return gguf.GGMLQuantizationType.F16
        return False

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        # embeddings.weight is only read for the semantic rows, see modify_tensors()
        if name.startswith(("fast_", "codebook_embeddings")) or name == "embeddings.weight":
            return item
        return None  # the slow AR is the text model

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        g = self.global_config

        if name.startswith("a.gen.wav."):
            yield (name, data_torch)  # codec, named by generate_extra_tensors()
            return
        if name == "embeddings.weight":
            return  # the text model has it

        if name == "codebook_embeddings.weight":
            # fold the semantic rows into codebook 0, see the header
            begin, n = g["semantic_begin_id"], g["codebook_size"]
            table = data_torch.float()
            sem = self.model_tensors["embeddings.weight"]()[begin:begin + n].float()
            yield (self.format_tensor_name(T.A_GEN_AUDIO_EMBD), torch.cat([table[:n] + sem, table[n:]], dim=0))
            return
        if name == "fast_embeddings.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_EMBD), data_torch)
            return
        if name == "fast_output.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_HEAD), data_torch)
            return
        if name == "fast_norm.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_OUTPUT_NORM), data_torch)
            return

        assert name.startswith("fast_layers.") and bid is not None, name
        key = name[len(f"fast_layers.{bid}."):]
        if key == "attention.wqkv.weight":
            q, k, v = _split_qkv(data_torch, g["fast_n_head"], g["fast_n_local_heads"], g["fast_head_dim"])
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_Q, bid), q)
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_K, bid), k)
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_V, bid), v)
            return
        yield (self.format_tensor_name(self._FAST_MAP[key], bid), data_torch)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        g = self.global_config
        T = gguf.MODEL_TENSOR
        logger.info(f"gguf: loading the codec from {g['codec_filename']}")
        sd = torch.load(self.dir_model / g["codec_filename"], map_location="cpu", weights_only=True)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        if any("generator." in k for k in sd):
            sd = {k.replace("generator.", ""): v for k, v in sd.items() if "generator." in k}

        def get(name: str) -> Tensor:
            return sd[name].float()

        def wn(prefix: str) -> Tensor:
            # both weight norm flavours of the checkpoint
            if prefix + ".parametrizations.weight.original0" in sd:
                return _weight_norm(get(prefix + ".parametrizations.weight.original0"), get(prefix + ".parametrizations.weight.original1"))
            return _weight_norm(get(prefix + ".weight_g"), get(prefix + ".weight_v"))

        def snake(prefix: str) -> tuple[Tensor, Tensor]:
            # Snake1d: x + 1/(alpha + 1e-9) * sin(alpha * x)^2, the code2wav snake takes the reciprocal as beta
            alpha = get(prefix + ".alpha").reshape(-1)
            return alpha, 1.0 / (alpha + 1e-9)

        def projected_codebook(prefix: str) -> Tensor:
            # [vocab, 8] codes -> [vocab, 1024] rows of out_proj(code) + bias
            w = wn(prefix + ".out_proj").squeeze(-1)  # [1024, 8]
            return get(prefix + ".codebook.weight") @ w.t() + get(prefix + ".out_proj.bias")

        q = "quantizer"
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_FIRST_CB), projected_codebook(f"{q}.semantic_quantizer.quantizers.0"))
        n_rest = sum(1 for k in sd if k.startswith(f"{q}.quantizer.quantizers.") and k.endswith(".codebook.weight"))
        rest = [projected_codebook(f"{q}.quantizer.quantizers.{i}") for i in range(n_rest)]
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_REST_CB), torch.stack(rest, dim=0))

        # post_module: window transformer, wqkv split, q/k moved to the NEOX layout
        n_head, n_head_kv = g["codec_post_n_head"], g["codec_post_n_local_heads"]
        head_dim = 64
        n_tfm = sum(1 for k in sd if k.startswith(f"{q}.post_module.layers.") and k.endswith(".attention.wqkv.weight"))
        for bid in range(n_tfm):
            p = f"{q}.post_module.layers.{bid}."
            wq, wk, wv = _split_qkv(get(p + "attention.wqkv.weight"), n_head, n_head_kv, head_dim)
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_ATTN_Q, bid), _pairs_to_halves(wq, head_dim))
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_ATTN_K, bid), _pairs_to_halves(wk, head_dim))
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_ATTN_V, bid), wv)
            for key, tensor_id in self._TFM_MAP.items():
                yield (self.format_tensor_name(tensor_id, bid), get(p + key))
        yield (self.format_tensor_name(T.A_GEN_WAV_TFM_OUTPUT_NORM), get(f"{q}.post_module.norm.weight"))

        # upsample: 2x (causal ConvTranspose1d + ConvNeXt block), same layout as qwen3-tts
        up_map = {
            "0.conv.weight":        (T.A_GEN_WAV_UP_CONV,   ".weight"),
            "0.conv.bias":          (T.A_GEN_WAV_UP_CONV,   ".bias"),
            "1.dwconv.conv.weight": (T.A_GEN_WAV_UP_DWCONV, ".weight"),
            "1.dwconv.conv.bias":   (T.A_GEN_WAV_UP_DWCONV, ".bias"),
            "1.norm.weight":        (T.A_GEN_WAV_UP_NORM,   ".weight"),
            "1.norm.bias":          (T.A_GEN_WAV_UP_NORM,   ".bias"),
            "1.pwconv1.weight":     (T.A_GEN_WAV_UP_PW1,    ".weight"),
            "1.pwconv1.bias":       (T.A_GEN_WAV_UP_PW1,    ".bias"),
            "1.pwconv2.weight":     (T.A_GEN_WAV_UP_PW2,    ".weight"),
            "1.pwconv2.bias":       (T.A_GEN_WAV_UP_PW2,    ".bias"),
            "1.gamma":              (T.A_GEN_WAV_UP_GAMMA,  ""),
        }
        for bid in range(2):
            for key, (tensor_id, suffix) in up_map.items():
                yield (self.format_tensor_name(tensor_id, bid, suffix=suffix), get(f"{q}.upsample.{bid}.{key}"))

        # DAC style decoder: decoder.model.0 conv, 1..4 blocks, 5 snake, 6 conv, then tanh (in the graph)
        d = "decoder.model"
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_ENTRY, suffix=".weight"), wn(f"{d}.0.conv"))
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_ENTRY, suffix=".bias"), get(f"{d}.0.conv.bias"))
        for bid in range(4):
            b = f"{d}.{bid + 1}.block"
            a, beta = snake(f"{b}.0")
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_SNAKE, bid, suffix=".alpha"), a)
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_SNAKE, bid, suffix=".beta"), beta)
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_CONV, bid, suffix=".weight"), wn(f"{b}.1.conv"))
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_CONV, bid, suffix=".bias"), get(f"{b}.1.conv.bias"))
            for xid in range(3):
                r = f"{b}.{xid + 2}.block"
                for tensor_id, snake_idx, conv_idx in ((T.A_GEN_WAV_DAC_RES_ACT1, 0, None), (T.A_GEN_WAV_DAC_RES_CONV1, None, 1),
                                                       (T.A_GEN_WAV_DAC_RES_ACT2, 2, None), (T.A_GEN_WAV_DAC_RES_CONV2, None, 3)):
                    base = gguf.TENSOR_NAMES[tensor_id].format(bid=bid, xid=xid)
                    if snake_idx is not None:
                        a, beta = snake(f"{r}.{snake_idx}")
                        yield (base + ".alpha", a)
                        yield (base + ".beta", beta)
                    else:
                        yield (base + ".weight", wn(f"{r}.{conv_idx}.conv"))
                        yield (base + ".bias", get(f"{r}.{conv_idx}.conv.bias"))
        a, beta = snake(f"{d}.5")
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_SNAKE, suffix=".alpha"), a)
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_SNAKE, suffix=".beta"), beta)
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_CONV, suffix=".weight"), wn(f"{d}.6.conv"))
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_CONV, suffix=".bias"), get(f"{d}.6.conv.bias"))
