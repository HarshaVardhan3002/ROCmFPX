"""Phase 1 MoE arm: quantize ONLY the routed experts of a Qwen3.5/3.6-MoE model (fused gate_up_proj / down_proj),
everything else stays as in the template (F16, embd/output Q6_K-expanded). Per-expert Hessians from a sequential
transformers forward, shrunk toward the layer Hessian: H_e = (S_e + n0 * Hbar) / (n_e + n0). Batched GPTQ over experts.

python -I rq_moe.py --gguf-py DIR --hf MODEL_DIR --template T.gguf --calib TXT --fmt S40|FP3|I4|R2|R3
                    --enc diag|gptq --n0 2048 --out OUT.gguf
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq  # noqa: E402


def gptq_b(W, H, qgroup, g, blocksize=128, damp=0.01):
    """Batched GPTQ. W (B,R,C), H (B,C,C). qgroup(Wcols (B,R,g), c) -> fn(col (B,R)) -> dequantized col."""
    B, R, C = W.shape
    W = W.clone()
    H = H.clone()
    d = H.diagonal(dim1=1, dim2=2)
    dead = d <= 0
    W = W * (~dead)[:, None, :]
    idx = torch.arange(C, device=W.device)
    H[:, idx, idx] = torch.where(dead, torch.ones_like(d), d)
    mean_d = H.diagonal(dim1=1, dim2=2).mean(1)
    for k in range(6):
        Hd = H + (damp * 10 ** k) * mean_d[:, None, None] * torch.eye(C, device=W.device)[None]
        L, info = torch.linalg.cholesky_ex(Hd)
        if (info == 0).all():
            break
        H = torch.where((info != 0)[:, None, None], Hd, H)
    Hinv = torch.cholesky_inverse(L)
    del L, Hd, H
    U = torch.linalg.cholesky(Hinv, upper=True)
    del Hinv
    Q = torch.zeros_like(W)
    rqf = None
    for i1 in range(0, C, blocksize):
        i2 = min(i1 + blocksize, C)
        W1 = W[:, :, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Ui = U[:, i1:i2, i1:i2]
        for j in range(i2 - i1):
            c = i1 + j
            if c % g == 0:
                rqf = qgroup(W1[:, :, j:j + g], c)
            w = W1[:, :, j]
            q = rqf(w)
            Q1[:, :, j] = q
            e = (w - q) / Ui[:, j, j][:, None]
            W1[:, :, j:] -= e[:, :, None] * Ui[:, j, j:][:, None, :]
            E1[:, :, j] = e
        Q[:, :, i1:i2] = Q1
        if i2 < C:
            W[:, :, i2:] -= E1 @ U[:, i1:i2, i2:]
    return Q


def ts_exp_b(W, ql):
    return [rq.ts_exponent(W[e], ql) for e in range(W.shape[0])]


def quant_experts(W, H, fmt, enc):
    """W (E,R,C) fp32, H (E,C,C) -> dequantized (E,R,C)."""
    E, R, C = W.shape
    hd = H.diagonal(dim1=1, dim2=2).clamp_min(0)          # per-expert column importance (imatrix analogue)
    if fmt in ("R2", "R3"):
        L = rq.RLEV[fmt]
        w8 = hd.clamp_min(1e-12)
        sig = torch.sqrt((W * W * w8[:, None, :]).sum(2) / w8.sum(1, keepdim=True))
        s0 = (rq.RINIT[fmt] * sig).clamp_min(1e-8)
        best, best_err = s0.clone(), torch.full_like(s0, float("inf"))
        for f in torch.linspace(0.6, 1.6, 41, device=W.device):
            sf = s0 * f
            err = (((W - rq.rround(W, sf[:, :, None], L)) ** 2) * w8[:, None, :]).sum(2)
            m = err < best_err
            best, best_err = torch.where(m, sf, best), torch.where(m, err, best_err)
        s = best
        if enc == "gptq":
            Q = gptq_b(W, H, lambda xg, c: (lambda col: rq.rround(col, s, L)), g=C)
        else:
            Q = rq.rround(W, s[:, :, None], L)
        Cc = Q / s[:, :, None]
        CH = Cc @ H
        num, den = (CH * W).sum(2), (CH * Cc).sum(2).clamp_min(1e-20)
        s2 = torch.where(den > 0, num / den, s)
        return (Cc * s2[:, :, None]).half().float()
    g, dec, qmax, ql = rq.BLOCK[fmt]
    ks = torch.tensor([2.0 ** k for k in ts_exp_b(W, ql)], device=W.device)[:, None, None]
    x = W * ks
    sigma2 = (x * x).mean(2, keepdim=True)
    if enc == "diag":
        wts = hd[:, None, :] * torch.sqrt(sigma2 + x * x)
        s = rq.choose_scales(x.reshape(-1, g), wts.reshape(-1, g), dec, qmax)
        q = dec(x.reshape(-1, g), s[:, None]).reshape(E, R, C)
    else:
        def qgroup(xg, c):
            wts = hd[:, None, c:c + g] * torch.sqrt(sigma2 + xg * xg)
            s = rq.choose_scales(xg.reshape(-1, g), wts.reshape(-1, g), dec, qmax).reshape(E, R)
            return lambda col: dec(col, s)
        q = gptq_b(x, H, qgroup, g)
    return q / ks


def main():
    ap = argparse.ArgumentParser()
    for a in ["gguf-py", "hf", "template", "calib", "out"]:
        ap.add_argument("--" + a, required=True)
    ap.add_argument("--fmt", required=True, choices=["S40", "FP3", "I4", "R2", "R3"])
    ap.add_argument("--enc", default="gptq", choices=["diag", "gptq"])
    ap.add_argument("--n0", type=float, default=2048.0)
    ap.add_argument("--nchunks", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--qprefix", type=int, default=1)
    ap.add_argument("--search", default="exhaustive", choices=["cpp", "exhaustive"])
    ap.add_argument("--rw", action="store_true", help="weight each routed token by its squared routing weight")
    ap.add_argument("--window", default="16,8", help="scale-index window below,above nearest(max/qmax); empty = all 126")
    ap.add_argument("--stats", help="write per-expert token counts (json)")
    ap.add_argument("--smoke", type=int, default=0, help="stop after N layers, no export")
    args = ap.parse_args()
    sys.path.insert(0, args.gguf_py)
    import gguf
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda")
    rq.UE = rq.ue4m3_table().to(dev)
    rq.SEARCH = args.search
    rq.WINDOW = tuple(int(v) for v in args.window.split(",")) if args.window else None
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.hf)
    ids = tok(Path(args.calib).read_text(encoding="utf-8", errors="ignore"), return_tensors="pt").input_ids[0]
    n = min(args.nchunks, ids.numel() // args.ctx)
    calib = ids[: n * args.ctx].reshape(n, args.ctx)
    assert n > 0, 'no calibration chunks (tokenizer or calib file problem)'
    model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16).to(dev).eval()
    nl = model.config.get_text_config().num_hidden_layers
    layers = next(m for m in model.modules() if isinstance(m, torch.nn.ModuleList) and len(m) == nl)
    print(f"{type(model).__name__}: {nl} layers, calib {n}x{args.ctx}", flush=True)

    inps, kw = [], {}

    class Stop(Exception):
        pass

    def catch(mod, a, k):
        inps.append(a[0] if a else k["hidden_states"])
        if not kw:
            kw.update({kk: vv for kk, vv in k.items() if kk != "hidden_states"})
        raise Stop

    h = layers[0].register_forward_pre_hook(catch, with_kwargs=True)
    with torch.no_grad():
        for i in range(0, n, 8):
            try:
                model(input_ids=calib[i:i + 8].to(dev), use_cache=False)
            except Stop:
                pass
    h.remove()
    print(f"captured {len(inps)} batches; kwargs {sorted(kw)}", flush=True)

    def run_layer(layer, x):
        o = layer(x, **kw)
        return o[0] if isinstance(o, tuple) else o

    results, stats = {}, {}
    spill = Path(args.out + ".spill"); spill.mkdir(parents=True, exist_ok=True)
    for li, layer in enumerate(layers):
        ex = layer.mlp.experts
        E, I2, Hd = ex.gate_up_proj.shape
        I = I2 // 2
        S_gu = torch.zeros(E, Hd, Hd, device=dev)
        S_d = torch.zeros(E, I, I, device=dev)
        cnt = torch.zeros(E, device=dev)
        sw2 = torch.zeros(E, device=dev)          # sum of squared routing weights per expert (--rw)
        Hbar_gu = torch.zeros(Hd, Hd, device=dev)
        n_tok = [0]

        def hook(mod, a, k):
            hs = (a[0] if a else k["hidden_states"]).float()
            idx = a[1] if len(a) > 1 else k["top_k_index"]
            tkw = (a[2] if len(a) > 2 else k["top_k_weights"]).float() if args.rw else None
            Hbar_gu.addmm_(hs.t(), hs)
            n_tok[0] += hs.shape[0]
            for e in torch.unique(idx).tolist():
                sel = idx == e
                tok_i = sel.any(1).nonzero().squeeze(1)
                x = hs[tok_i]
                cnt[e] += x.shape[0]
                gu = x @ mod.gate_up_proj[e].float().t()
                hmid = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
                if tkw is not None:      # expert output is scaled by its routing weight: weight the error by w^2
                    wt = (tkw * sel).sum(1)[tok_i]
                    sw2[e] += (wt * wt).sum()
                    x, hmid = x * wt[:, None], hmid * wt[:, None]
                S_gu[e].addmm_(x.t(), x)
                S_d[e].addmm_(hmid.t(), hmid)

        hk = ex.register_forward_pre_hook(hook, with_kwargs=True)
        with torch.no_grad():
            for x in inps:
                run_layer(layer, x)
        hk.remove()
        Hbar_gu /= max(n_tok[0], 1)
        if args.rw:   # per-expert weighted means, rescaled to token counts so the shrinkage n0 keeps its meaning
            f = (cnt / sw2.clamp_min(1e-12))[:, None, None]
            S_gu, S_d = S_gu * f, S_d * f
            Hbar_gu = S_gu.sum(0) / cnt.sum().clamp_min(1)
        Hbar_d = S_d.sum(0) / cnt.sum().clamp_min(1)
        n0 = args.n0
        H_gu = (S_gu + n0 * Hbar_gu[None]) / (cnt + n0).clamp_min(1e-6)[:, None, None]
        H_d = (S_d + n0 * Hbar_d[None]) / (cnt + n0).clamp_min(1e-6)[:, None, None]
        del S_gu, S_d
        stats[li] = cnt.long().tolist()
        Wgu = ex.gate_up_proj.data.float()
        Qg = quant_experts(Wgu[:, :I].contiguous(), H_gu, args.fmt, args.enc)
        Qu = quant_experts(Wgu[:, I:].contiguous(), H_gu, args.fmt, args.enc)
        del H_gu
        Qd = quant_experts(ex.down_proj.data.float(), H_d, args.fmt, args.enc)
        del H_d
        for nm, Q in (("gate", Qg), ("up", Qu), ("down", Qd)):   # spill to disk: 1.6 GB/layer would not fit next to the model
            key = f"blk.{li}.ffn_{nm}_exps.weight"
            results[key] = spill / f"{key}.npy"
            np.save(results[key], Q.half().cpu().numpy())
        if args.qprefix:
            ex.gate_up_proj.data.copy_(torch.cat([Qg, Qu], 1).to(torch.bfloat16))
            ex.down_proj.data.copy_(Qd.to(torch.bfloat16))
        with torch.no_grad():
            inps = [run_layer(layer, x) for x in inps]
        c = cnt.cpu()
        print(f"layer {li} done {time.time() - t0:.0f}s tokens/expert min {int(c.min())} median {int(c.median())} "
              f"max {int(c.max())} (<{int(n0)}: {int((c < n0).sum())})", flush=True)
        if args.smoke and li + 1 >= args.smoke:
            for k, v in results.items():
                v = torch.from_numpy(np.load(v)).float()
                print("SMOKE", k, tuple(v.shape), float(v.abs().mean()), bool(torch.isfinite(v).all()), flush=True)
            print("SMOKE_OK", flush=True)
            return
    if args.stats:
        import json
        Path(args.stats).write_text(json.dumps(stats))
    del model, inps
    torch.cuda.empty_cache()

    tr = gguf.GGUFReader(args.template)
    w = gguf.GGUFWriter(args.out, tr.fields["general.architecture"].contents(), endianess=tr.endianess)
    for key, field in tr.fields.items():
        if key.startswith(("GGUF.", "split.")) or key == "general.architecture":
            continue
        sub = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(key, field.contents(), field.types[0], sub_type=sub)
    for t in tr.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.n_bytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    checked = 0
    for t in tr.tensors:
        if t.name in results:
            a = np.load(results[t.name], mmap_mode="r")
            assert t.tensor_type == gguf.GGMLQuantizationType.F16 and a.shape == tuple(t.data.shape), (t.name, a.shape, t.data.shape)
            if checked < 3:   # name/layout mapping check: quantized vs original should correlate strongly
                ref = np.asarray(t.data, dtype=np.float32)[:4]
                cc = np.corrcoef(ref.reshape(-1), a[:4].astype(np.float32).reshape(-1))[0, 1]
                print(f"MAPCHECK {t.name} corr {cc:.4f}", flush=True)
                assert cc > 0.5, "expert tensor mapping looks wrong"
                checked += 1
            w.write_tensor_data(a)
        else:
            w.write_tensor_data(t.data, tensor_endianess=tr.endianess)
    w.close()
    import shutil
    shutil.rmtree(spill, ignore_errors=True)
    print(f"RQMOE_DONE {args.fmt} {args.enc} n0={args.n0} {len(results)} tensors {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
