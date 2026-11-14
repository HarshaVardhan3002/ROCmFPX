#!/usr/bin/env python3
"""Drop routed experts from an Inkling 2 GGUF (and, optionally, from its imatrix).

  prune_experts.py stats.tsv KEEP --model in.gguf out.gguf [--imatrix in.imatrix.gguf out.imatrix.gguf]
                   [--score reap|freq|weight] [--keep-file keep.json]

stats.tsv comes from inkling2-expert-stats. Every layer keeps the same number of experts (llama.cpp has one
expert_count), the KEEP highest scoring ones, in their original order:
  reap   = sum(w * ||expert output||) / count   (REAP saliency, default)
  freq   = count
  weight = sum(w)
Experts never routed score 0. Tensors sliced along the expert axis: ffn_{gate,up,down}_exps.weight/.scale,
ffn_gate_inp.weight, exp_probs_b.bias; quantized data is sliced byte-wise (rows of one expert are contiguous).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402

EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                   "ffn_gate_exps.scale", "ffn_up_exps.scale", "ffn_down_exps.scale",
                   "ffn_gate_inp.weight", "exp_probs_b.bias")


def load_stats(path):
    rows = np.loadtxt(path, comments="#", skiprows=2)
    n_layer = int(np.max(rows[:, 0])) + 1
    n_expert = int(np.max(rows[:, 1])) + 1
    get = lambda c: rows[:, c].reshape(n_layer, n_expert)  # noqa: E731
    return get(2), get(3), get(4)


def choose(stats_path, keep, score):
    cnt, sw, swn = load_stats(stats_path)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = {"reap": np.where(cnt > 0, swn / cnt, 0.0), "freq": cnt, "weight": sw}[score]
    sets = {}
    for il in range(s.shape[0]):
        order = np.argsort(-s[il], kind="stable")
        sets[il] = sorted(int(e) for e in order[:keep])
    return sets, cnt


def layer_of(name):
    p = name.split(".")
    return int(p[1]) if len(p) > 2 and p[0] == "blk" and p[1].isdigit() else None


def copy_fields(r, w, skip, override):
    for key, field in r.fields.items():
        if key.startswith(("GGUF.", "split.")) or key in skip:
            continue
        if key in override:
            continue
        sub = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(key, field.contents(), field.types[0], sub_type=sub)
    for key, (val, vt) in override.items():
        w.add_key_value(key, val, vt)


def rewrite(src, dst, sets, n_expert, is_expert_tensor, arch_key_override):
    r = gguf.GGUFReader(src)
    arch_f = r.fields.get("general.architecture")
    arch = arch_f.contents() if arch_f is not None else None
    w = gguf.GGUFWriter(dst, arch or "", endianess=r.endianess)  # imatrix files have no architecture; "" writes none
    copy_fields(r, w, {"general.architecture"} if arch else set(), arch_key_override(arch))
    plan = []
    for t in r.tensors:
        il = layer_of(t.name)
        data = t.data
        if il is not None and is_expert_tensor(t.name):
            if data.shape[0] != n_expert:
                raise SystemExit(f"{t.name}: expert axis {data.shape[0]} != {n_expert}")
            idx = np.asarray(sets[il])
            shape = (len(idx),) + tuple(data.shape[1:])
            nbytes = int(t.n_bytes // n_expert * len(idx))
            plan.append((t, idx))
        else:
            shape, nbytes, idx = tuple(data.shape), int(t.n_bytes), None
            plan.append((t, None))
        w.add_tensor_info(t.name, shape, data.dtype, nbytes, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t, idx in plan:
        w.write_tensor_data(t.data if idx is None else np.ascontiguousarray(t.data[idx]), tensor_endianess=r.endianess)
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("keep", type=int)
    ap.add_argument("--model", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--imatrix", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--score", choices=["reap", "freq", "weight"], default="reap")
    ap.add_argument("--keep-file")
    a = ap.parse_args()

    sets, cnt = choose(a.stats, a.keep, a.score)
    n_expert = cnt.shape[1]
    tot = cnt.sum(1, keepdims=True)
    kept_share = np.array([cnt[il, sets[il]].sum() for il in sets]) / tot[:, 0]
    print(f"keep {a.keep}/{n_expert} per layer ({a.score}); routed-token share kept: "
          f"min {kept_share.min():.4f} mean {kept_share.mean():.4f}")
    if a.keep_file:
        Path(a.keep_file).write_text(json.dumps({"score": a.score, "keep": a.keep, "sets": sets,
                                                 "kept_share": kept_share.tolist()}))
    if a.model:
        def ovr(arch):
            return {f"{arch}.expert_count": (a.keep, gguf.GGUFValueType.UINT32)}
        rewrite(a.model[0], a.model[1], sets, n_expert,
                lambda n: n.split(".", 2)[2] in EXPERT_SUFFIXES, ovr)
        print("wrote", a.model[1])
    if a.imatrix:
        # imatrix entries of the merged expert tensors: <tensor>.in_sum2 / .in_sum [n_embd, n_expert], .counts [1, n_expert]
        # (.stats is a per-tensor summary and stays as it is)
        def is_exp(n):
            return any(n.endswith(s + sfx) for s in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
                       for sfx in (".in_sum2", ".in_sum", ".counts"))
        rewrite(a.imatrix[0], a.imatrix[1], sets, n_expert, is_exp, lambda arch: {})
        print("wrote", a.imatrix[1])


if __name__ == "__main__":
    main()
