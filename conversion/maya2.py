from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, TextModel, gguf, logger

# Maya2 (maya-research/maya2-native) is a Qwen3 backbone that predicts Mimi codebook 0,
# a depth decoder that predicts codebooks 1..31, and the kyutai/mimi decoder.
#
# Tricks being used to support this model via existing llama.cpp code paths:
# - the text lm_head is not used, the cb0 head (2048 codes + AEOS) is the output tensor
#   the logits are padded with -inf up to n_vocab at inference time
# - the named voices use the null speaker anchor, which is added to the <|audio|> sentinel
#   this sum is appended to the embedding table as the token <|audio_null_spk|>
# - with one null speaker memory token, the depth cross-attention output is a constant per layer
#   it is computed here and stored as one column per layer
# - the Mimi decoder is not in the checkpoint, it is downloaded from kyutai/mimi
#
# pipeline stage mapping:
#   backbone           --> mapped to normal libllama text model (autoregressive)
#   depth decoder      --> MTMD_GEN_PROCESS_TYPE_GEN_CODE (+ the frame feedback embedding)
#   mimi decoder       --> MTMD_GEN_PROCESS_TYPE_GEN_WAV

_MIMI_REPO = "kyutai/mimi"
_MIMI_REVISION = "89091b3e466eb6a9d11e537bf26b144f194978f7"

_AUDIO_TOKEN = "<|audio|>"
_AUDIO_NULL_SPK_TOKEN = "<|audio_null_spk|>"

# indices into mimi decoder.layers for stage i, see MimiDecoder
_DEC_SCALE_IDX = lambda i: 2 + 3 * i  # noqa: E731
_DEC_RES_IDX   = lambda i: 3 + 3 * i  # noqa: E731
_N_SEANET_STAGES = 4


def _is_maya2(dir_model: Path) -> bool:
    return (dir_model / "maya3_config.json").is_file() and (dir_model / "backbone" / "config.json").is_file()


def _mimi_config() -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(_MIMI_REPO, "config.json", revision=_MIMI_REVISION), encoding="utf-8") as f:
        return json.load(f)


@ModelBase.register_hparams_loader(_is_maya2)
def _load_maya2_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected a maya2 checkpoint, the backbone config is in backbone/")
    with open(dir_model / "backbone" / "config.json", encoding="utf-8") as f:
        hparams = json.load(f)
    hparams["architectures"] = ["Maya2Model"]
    # one extra row for <|audio_null_spk|>
    hparams["vocab_size"] = hparams["vocab_size"] + 1
    # the mimi decoder is the gen-audio part of the mmproj
    hparams["audio_config"] = _mimi_config()
    return hparams


def _null_anchor(tensors: dict[str, Callable[[], Tensor]]) -> Tensor:
    # see SpeakerAnchor.forward() with the null path: fc2(gelu(fc1(null_vec)))
    x = tensors["speaker_anchor.null_vec"]().float()
    x = torch.nn.functional.linear(x, tensors["speaker_anchor.fc1.weight"]().float(), tensors["speaker_anchor.fc1.bias"]().float())
    x = torch.nn.functional.gelu(x)
    return torch.nn.functional.linear(x, tensors["speaker_anchor.fc2.weight"]().float(), tensors["speaker_anchor.fc2.bias"]().float())


@ModelBase.register("Maya2Model")
# [TAG_HF_EXAMPLE_MISSING] the checkpoint keeps the backbone config in a subdir, not supported here
class Maya2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MAYA2

    def get_vocab_base_pre(self, tokenizer) -> str:
        del tokenizer
        # same pre-tokenizer regex as qwen3.5, the hash differs because the vocab differs
        return "qwen35"

    def set_vocab(self):
        dir_model = self.dir_model
        self.dir_model = dir_model / "backbone"
        try:
            tokens, toktypes, tokpre = self.get_vocab_base()
            # get_vocab_base() fills the extra row as [PAD], name it
            assert tokens[-1] == f"[PAD{len(tokens) - 1}]"
            tokens[-1] = _AUDIO_NULL_SPK_TOKEN
            toktypes[-1] = gguf.TokenType.CONTROL

            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)

            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
            special_vocab.add_to_gguf(self.gguf_writer)
        finally:
            self.dir_model = dir_model

        assert tokens.index(_AUDIO_TOKEN) == self._audio_token_id()
        self.gguf_writer.add_add_bos_token(False)
        # this is not a chat model
        self.gguf_writer.add_chat_template(None)

    def _audio_token_id(self) -> int:
        # tensors can be written before set_vocab(), so read the id from the tokenizer directly
        with open(self.dir_model / "backbone" / "tokenizer.json", encoding="utf-8") as f:
            added = json.load(f)["added_tokens"]
        return next(t["id"] for t in added if t["content"] == _AUDIO_TOKEN)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name == "backbone.cb0_head.weight" or name.startswith("speaker_anchor."):
            return item  # the anchor is folded into the embedding table, see modify_tensors()
        if not name.startswith("backbone.lm.model."):
            return None  # the depth decoder and the audio table go to the mmproj
        return super().filter_tensors((name[len("backbone.lm."):], gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "backbone.cb0_head.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch)
            return
        if name.startswith("speaker_anchor."):
            return

        if name == "model.embed_tokens.weight":
            # append the <|audio|> row plus the null speaker anchor, see compose_inputs()
            anchor = _null_anchor(self.model_tensors)
            row = data_torch[self._audio_token_id()].float() + anchor
            data_torch = torch.cat([data_torch, row.reshape(1, -1).to(data_torch.dtype)], dim=0)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Maya2Model")
# [TAG_HF_EXAMPLE_MISSING] the checkpoint keeps the backbone config in a subdir, not supported here
class Maya2MmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = False

    _DEPTH_BLK_MAP = {
        "n1":  gguf.MODEL_TENSOR.A_GEN_DEPTH_ATTN_NORM,
        "qkv": gguf.MODEL_TENSOR.A_GEN_DEPTH_ATTN_QKV,
        "o":   gguf.MODEL_TENSOR.A_GEN_DEPTH_ATTN_OUT,
        "n2":  gguf.MODEL_TENSOR.A_GEN_DEPTH_FFN_NORM,
        "f1":  gguf.MODEL_TENSOR.A_GEN_DEPTH_FFN_UP,
        "f2":  gguf.MODEL_TENSOR.A_GEN_DEPTH_FFN_DOWN,
    }
    _MIMI_TFM_MAP = {
        "input_layernorm":          gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_NORM,
        "post_attention_layernorm": gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_NORM,
        "self_attn.q_proj":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_Q,
        "self_attn.k_proj":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_K,
        "self_attn.v_proj":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_V,
        "self_attn.o_proj":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_OUT,
        "mlp.fc1":                  gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_UP,
        "mlp.fc2":                  gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_DOWN,
        "self_attn_layer_scale":    gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_SCALE,
        "mlp_layer_scale":          gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_SCALE,
    }

    def __init__(self, dir_model: Path, *args, **kwargs):
        from huggingface_hub import snapshot_download
        logger.info(f"gguf: downloading the mimi decoder {_MIMI_REPO}")
        self.dir_mimi = Path(snapshot_download(_MIMI_REPO, revision=_MIMI_REVISION,
                                               allow_patterns=["config.json", "model.safetensors"]))
        super().__init__(dir_model, *args, **kwargs)
        self.n_depth_layer = sum(1 for name in self.model_tensors if name.startswith("depth.cross.") and name.endswith(".kv.weight"))
        self.n_depth_level = sum(1 for name in self.model_tensors if name.startswith("depth.heads."))

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors = super().index_tensors(remote_hf_model_id=remote_hf_model_id)
        dir_model = self.dir_model
        self.dir_model = self.dir_mimi
        try:
            mimi = super().index_tensors()
        finally:
            self.dir_model = dir_model
        for name, gen in mimi.items():
            # the encoder is not used, the voices are named
            if name.startswith(("encoder", "downsample", "encoder_transformer")):
                continue
            tensors["mimi." + name] = gen
        return tensors

    def set_gguf_parameters(self):
        assert self.hparams_audio is not None
        mimi = self.hparams_audio
        self.gguf_writer.add_file_type(self.ftype)

        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.MAYA2_GEN)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        # these describe the mimi decoder transformer, clip.cpp reads the depth decoder from the tensor shapes
        self.gguf_writer.add_gen_audio_embedding_length(mimi["hidden_size"])
        self.gguf_writer.add_gen_audio_feed_forward_length(mimi["intermediate_size"])
        self.gguf_writer.add_gen_audio_block_count(mimi["num_hidden_layers"])
        self.gguf_writer.add_gen_audio_head_count(mimi["num_attention_heads"])
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(mimi["norm_eps"])
        self.gguf_writer.add_gen_audio_model_variant("maya2")

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del name, bid, n_dims
        # conv1d/conv1d_dw kernels must be F16, ggml_conv_1d(_dw) has no BF16 path
        if ".seanet." in new_name or new_name == "a.gen.wav.upsample.weight":
            return gguf.GGMLQuantizationType.F16
        # small and used as get_rows sources or broadcast adds
        if new_name.startswith(("a.gen.depth.level_embd", "a.gen.depth.cross", "a.gen.wav.quant.")):
            return gguf.GGMLQuantizationType.F32
        return False

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # block indices are parsed here
        T = gguf.MODEL_TENSOR

        if name.startswith("mimi."):
            yield from self._mimi_tensor(name[len("mimi."):], data_torch)
            return

        if name == "audio_in.table.weight":
            yield (self.format_tensor_name(T.A_GEN_AUDIO_EMBD), data_torch)
            return
        if name == "audio_in.norm.weight":
            yield (self.format_tensor_name(T.A_GEN_AUDIO_EMBD_NORM), data_torch)
            return

        if not name.startswith("depth."):
            return  # backbone, speaker anchor (folded into the vocab), aeos_in (not used)
        key = name[len("depth."):]

        simple = {
            "in_proj.weight":   (T.A_GEN_DEPTH_IN_PROJ,    ".weight"),
            "code_lo.weight":   (T.A_GEN_DEPTH_CODE_LO,    ".weight"),
            "code_up.weight":   (T.A_GEN_DEPTH_CODE_UP,    ".weight"),
            "level_emb.weight": (T.A_GEN_DEPTH_LEVEL_EMBD, ".weight"),
            "norm.weight":      (T.A_GEN_DEPTH_NORM,       ".weight"),
            "norm.bias":        (T.A_GEN_DEPTH_NORM,       ".bias"),
        }
        if key in simple:
            tensor, suffix = simple[key]
            yield (self.format_tensor_name(tensor, suffix=suffix), data_torch)
            return

        if key == "null_mem":
            yield (self.format_tensor_name(T.A_GEN_DEPTH_CROSS), self._cross_const(data_torch))
            return

        if key == "heads.0.weight":
            heads = [self.model_tensors[f"depth.heads.{i}.weight"]().float() for i in range(self.n_depth_level)]
            yield (self.format_tensor_name(T.A_GEN_DEPTH_HEAD), torch.stack(heads, dim=0))
            return

        if key.startswith("groups."):
            group, layer, rest = key[len("groups."):].split(".", 2)
            sub, suffix = rest.rsplit(".", 1)
            blk = int(group) * self.n_depth_layer + int(layer)
            yield (self.format_tensor_name(self._DEPTH_BLK_MAP[sub], blk, suffix="." + suffix), data_torch)
            return

        # heads.N (merged above), cross.* (folded into the constants), spk_proj_* (no reference audio)
        return

    def _cross_const(self, null_mem: Tensor) -> Tensor:
        # with one memory token the attention weight is 1, so the block adds o(v(mem)) whatever x is
        mem = null_mem.float().reshape(-1)
        d = mem.shape[0]
        cols = []
        for il in range(self.n_depth_layer):
            kv = self.model_tensors[f"depth.cross.{il}.kv.weight"]().float()
            o  = self.model_tensors[f"depth.cross.{il}.o.weight"]().float()
            v  = torch.nn.functional.linear(mem, kv[d:])
            cols.append(torch.nn.functional.linear(v, o))
        return torch.stack(cols, dim=0)  # [n_layer, d]

    def _mimi_tensor(self, name: str, data_torch: Tensor) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR

        if name.startswith("quantizer."):
            yield from self._mimi_quant_tensor(name[len("quantizer."):], data_torch)
            return

        if name == "upsample.conv.weight":
            yield (self.format_tensor_name(T.A_GEN_WAV_UPSAMPLE), data_torch)
            return

        if name.startswith("decoder_transformer.layers."):
            bid = int(name.split(".")[2])
            key = name.split(f"layers.{bid}.", 1)[1]
            sub, suffix = key.rsplit(".", 1)
            tensor = self._MIMI_TFM_MAP.get(sub)
            if tensor is not None:
                suffix = "weight" if suffix == "scale" else suffix
                yield (self.format_tensor_name(tensor, bid, suffix="." + suffix), data_torch)
            return

        if name.startswith("decoder.layers."):
            idx = int(name.split(".")[2])
            suffix = "." + name.rsplit(".", 1)[1]
            if idx == 0:
                yield (self.format_tensor_name(T.A_GEN_WAV_SEANET_CONV_IN, suffix=suffix), data_torch)
                return
            if idx == 3 * _N_SEANET_STAGES + 2:
                yield (self.format_tensor_name(T.A_GEN_WAV_SEANET_CONV_OUT, suffix=suffix), data_torch)
                return
            for stage in range(_N_SEANET_STAGES):
                if idx == _DEC_SCALE_IDX(stage):
                    yield (self.format_tensor_name(T.A_GEN_WAV_SEANET_SCALE_CONV, stage, suffix=suffix), data_torch)
                    return
                if idx == _DEC_RES_IDX(stage):
                    # block.1 is the dilated conv, block.3 the pointwise one (0 and 2 are ELU)
                    inner = int(name.split(".block.", 1)[1].split(".")[0])
                    tensor = T.A_GEN_WAV_SEANET_RES_CONV1 if inner == 1 else T.A_GEN_WAV_SEANET_RES_CONV2
                    yield (self.format_tensor_name(tensor, stage, suffix=suffix), data_torch)
                    return
        return

    def _mimi_quant_tensor(self, name: str, data_torch: Tensor) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        sem = "semantic_residual_vector_quantizer."
        ac  = "acoustic_residual_vector_quantizer."

        def codebook(prefix: str, i: int) -> Tensor:
            # see MimiEuclideanCodebook: embed = embed_sum / cluster_usage
            p = f"mimi.quantizer.{prefix}layers.{i}.codebook."
            usage = self.model_tensors[p + "cluster_usage"]().float().clamp(min=1e-5)
            return self.model_tensors[p + "embed_sum"]().float() / usage[:, None]

        if name == sem + "layers.0.codebook.embed_sum":
            yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_FIRST_CB), codebook(sem, 0))
            return
        if name == ac + "layers.0.codebook.embed_sum":
            n = sum(1 for k in self.model_tensors if k.startswith("mimi.quantizer." + ac) and k.endswith(".embed_sum"))
            yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_REST_CB), torch.stack([codebook(ac, i) for i in range(n)], dim=0))
            return
        if name == sem + "output_proj.weight":
            yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_FIRST_OUT), data_torch.squeeze(-1))
            return
        if name == ac + "output_proj.weight":
            yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_REST_OUT), data_torch.squeeze(-1))
            return
        # input_proj belongs to the encoder, the other codebook entries are merged above
        return
