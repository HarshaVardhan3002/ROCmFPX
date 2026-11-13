# build a tiny random Qwen3.5-MoE (same classes as Qwen3.6-35B-A3B) for a smoke test of rq_moe.py
import json, sys, torch
from transformers import AutoConfig, AutoModelForCausalLM
src, out = sys.argv[1], sys.argv[2]
cfg = AutoConfig.from_pretrained(src)
t = cfg.get_text_config()
t.hidden_size, t.num_hidden_layers, t.num_experts, t.moe_intermediate_size = 256, 4, 16, 64
t.shared_expert_intermediate_size, t.num_attention_heads, t.num_key_value_heads = 64, 4, 2
t.linear_num_value_heads, t.linear_num_key_heads, t.layer_types = 4, 2, ["linear_attention"] * 3 + ["full_attention"]
for k in ("mtp_num_hidden_layers",):
    if hasattr(t, k): setattr(t, k, 0)
torch.manual_seed(0)
m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
m.save_pretrained(out)
import shutil, os
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
    if os.path.exists(os.path.join(src, f)): shutil.copy(os.path.join(src, f), out)
print("TINY", type(m).__name__, sum(p.numel() for p in m.parameters()))
