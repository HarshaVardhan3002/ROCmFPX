"""Phase 1 research quantizer (dense Qwen3 first).

Simulates ROCmFPX formats exactly (S40 = Q2_0_ROCMFPX, FP3 = Q3_0_ROCMFPX, I4 = Q4_0_ROCMI4, incl. the UE4M3
scale search, imatrix * sqrt(sigma2 + x^2) weights and the per-tensor power-of-two scale rule) plus the flat
row-scaled ROCMR family (R2 levels 2c-3, R3 levels 2c-7). Encoders: diag (= C++ llama-quantize) and gptq
(error feedback through the inverse Hessian). Weights come from the F16 GGUF, Hessians from a transformers forward.
Output: an F16 GGUF (template with the C++-identical Q6_K embedding/output expansion) for stock KLD scoring.

python -I rq.py --gguf-py DIR --hf MODEL_DIR --f16 F16.gguf --template TEMPLATE.gguf --imatrix IM.gguf
                --calib TXT --fmt S40|FP3|I4|R2|R3 --enc diag|gptq [--transform none|macaq --rot ROT.gguf] --out OUT.gguf
"""
import argparse, json, math, sys, time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

# ---------------------------------------------------------------- UE4M3 table (ggml/rocmfpx/rocmfpx.c)
def ue4m3_table():
    t = [m * 2.0 ** -10 for m in range(8)]
    for e in range(1, 16):
        for m in range(8):
            if len(t) < 127:
                t.append((8 + m) * 2.0 ** (e - 11))
    return torch.tensor(t, dtype=torch.float32)

UE: torch.Tensor = torch.empty(0)  # set on device in main

# ---------------------------------------------------------------- code rounding, exactly as the C++ encoders
def dec_s40(x, s):            # x, s broadcastable; returns reconstructed value
    a = (x * (1.0 / s)).abs()
    mag = torch.where(a > 2.5, 4.0, 1.0)
    return torch.where(torch.signbit(x), -mag, mag) * s

def dec_fp3(x, s):
    a = (x * (1.0 / s)).abs()
    mag = torch.where(a <= 0.5, 0.0, torch.where(a <= 1.5, 1.0, torch.where(a <= 3.0, 2.0, 4.0)))
    return torch.where(x < 0, -mag, mag) * s

def dec_i4(x, s):
    v = x * (1.0 / s)
    q = torch.sign(v) * torch.floor(v.abs() + 0.5)        # lroundf: half away from zero
    return q.clamp(-8, 7) * s

def safe(dec):
    def f(x, s):
        s = torch.as_tensor(s, device=x.device)
        return torch.where(s > 0, dec(x, torch.where(s > 0, s, torch.ones_like(s))), torch.zeros_like(x))
    return f

dec_s40, dec_fp3, dec_i4 = safe(dec_s40), safe(dec_fp3), safe(dec_i4)

BLOCK = {"S40": (16, dec_s40, 4.0, 2), "FP3": (16, dec_fp3, 4.0, 2), "I4": (32, dec_i4, 7.0, 3)}  # group, dec, qmax, qmax_log2

WINDOW: tuple[int, ...] | None = None    # (below, above): restrict the exhaustive search to scale indices around nearest(max/qmax)
SEARCH = "cpp"   # "cpp": replay the C++ search order incl. its clip pruning; "exhaustive": true argmin

def nearest_e(target):
    """rocmfpx_nearest_scale_ue4m3: index in 1..126 nearest to target, ties -> lower; 0 for target <= 0."""
    t = UE[1:127]
    lo = torch.searchsorted(t, target.contiguous()).clamp(0, 125) + 1      # first idx with table >= target
    lo_s, hi_s = UE[(lo - 1).clamp(min=0)], UE[lo]
    e = torch.where((lo > 1) & (target - lo_s <= hi_s - target), lo - 1, lo)
    return torch.where(target > 0, e, torch.zeros_like(e))

def choose_scales(x, w, dec, qmax, chunk=32768):
    """x, w: (G, n) groups and MSE weights -> (G,) UE4M3 scales, as rocmfpx_choose_scale_*_mse."""
    out = torch.empty(x.shape[0], device=x.device)
    for i in range(0, x.shape[0], chunk):
        xs, ws = x[i:i + chunk], w[i:i + chunk]
        ax = xs.abs()
        mx = ax.amax(1)
        if WINDOW and SEARCH == "exhaustive":     # argmin over a window of scale indices around nearest(max/qmax)
            st = nearest_e(mx / qmax).clamp(1, 126)
            idx = (st[:, None] + torch.arange(-WINDOW[0], WINDOW[1] + 1, device=x.device)[None]).clamp(1, 126)
            err = (ws[:, None, :] * (xs[:, None, :] - dec(xs[:, None, :], UE[idx][:, :, None])) ** 2).sum(-1)
            be = idx.gather(1, err.argmin(1, keepdim=True)).squeeze(1)
            s = UE[be]
            out[i:i + chunk] = torch.where(mx > 0, s, torch.zeros_like(s))
            continue
        err = (ws[:, None, :] * (xs[:, None, :] - dec(xs[:, None, :], UE[None, :, None])) ** 2).sum(-1)  # (g,127)
        err[:, 0] = float("inf")
        if SEARCH == "exhaustive":
            be = err.argmin(1)
        else:
            maw = torch.where(ax == mx[:, None], ws, torch.zeros_like(ws)).amax(1)
            start = nearest_e(mx / qmax)
            best = torch.full_like(mx, float("inf"))
            be = start.clone()
            done = torch.zeros_like(mx, dtype=torch.bool)
            ar = torch.arange(xs.shape[0], device=x.device)
            for d in range(126):
                e0 = start - d
                v0 = (~done) & (e0 >= 1)
                e0c = e0.clamp(1, 126)
                clip = mx - qmax * UE[e0c]
                prune = v0 & (clip > 0) & (maw * clip * clip > best)
                done |= prune
                t0 = v0 & ~prune
                r0 = err[ar, e0c]
                u = t0 & ((r0 < best) | ((r0 == best) & (e0c < be)))
                best, be = torch.where(u, r0, best), torch.where(u, e0c, be)
                if d:
                    e1 = start + d
                    v1 = e1 <= 126
                    e1c = e1.clamp(1, 126)
                    r1 = err[ar, e1c]
                    u = v1 & ((r1 < best) | ((r1 == best) & (e1c < be)))
                    best, be = torch.where(u, r1, best), torch.where(u, e1c, be)
        s = UE[be]
        out[i:i + chunk] = torch.where(mx > 0, s, torch.zeros_like(s))
    return out

def mse_weights(x, qw):
    """x (rows, k) already tensor-scaled; qw (k,) imatrix -> qw * sqrt(sigma2_row + x^2)."""
    sigma2 = (x * x).mean(1, keepdim=True)
    return qw[None, :] * torch.sqrt(sigma2 + x * x)

def ts_exponent(w, qmax_log2):
    """src/llama-quant.cpp rocmfpx_block_hist::exponent over blocks of 32."""
    m = w.reshape(-1, 32).abs().amax(1)
    m = m[(m > 0) & torch.isfinite(m)]
    if m.numel() == 0:
        return 0
    _, e = torch.frexp(m)
    e = e.clamp(-64, 63)
    hist = torch.bincount(e.long() + 64, minlength=128).cpu().numpy()
    total, acc, e_lo, e_hi = hist.sum(), 0, 0, 0
    for i in range(128):
        if hist[i] == 0:
            continue
        if acc * 100 < total:
            e_lo = i - 64
        acc += hist[i]
        e_hi = i - 64
    k_need = qmax_log2 - 5 - e_lo
    k_cap = min(5 + qmax_log2 - e_hi, 14 - e_hi)
    return int(min(max(min(k_need, k_cap), 0), 30))

# ---------------------------------------------------------------- block formats (S40 / FP3 / I4)
LAST = {}   # q (tensor-scaled domain) and k of the last block-format quantization, for real-file packing

def e_index(s):
    idx = torch.searchsorted(UE, s.contiguous()).clamp(0, 126)
    return idx, UE[idx] == s

def group_scale(a, ratios):
    """a = |q| of (N, g) groups -> (N,) e index whose scale reproduces the values with codes in `ratios`."""
    mn = torch.where(a > 0, a, torch.full_like(a, float("inf"))).amin(1)
    zero = torch.isinf(mn)
    best = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
    found = zero.clone()
    for div in (1.0, 2.0, 4.0):
        s = torch.where(zero, torch.zeros_like(mn), mn / div)
        idx, ok = e_index(s)
        r = a / torch.where(s > 0, s, torch.ones_like(s))[:, None]
        okr = ok & torch.stack([(r == v) for v in ratios]).any(0).all(1)
        take = okr & ~found
        best, found = torch.where(take, idx, best), found | take
    assert bool(found.all()), "could not recover block scales"
    return best

def pack_real(fmt, q):
    """Pack tensor-scaled dequantized values into ROCmFPX block bytes (same layout as ggml/rocmfpx/rocmfpx.c)."""
    rows, cols = q.shape
    g16 = q.reshape(-1, 16)
    a, neg = g16.abs(), torch.signbit(g16)
    if fmt == "S40":
        e = group_scale(a, (1.0, 4.0))
        s = UE[e][:, None]
        big = a > 2 * s
        code = torch.where(neg, torch.where(big, 0, 1), torch.where(big, 3, 2)).to(torch.uint8)
        c = code.reshape(-1, 4, 4).to(torch.int32)
        qs = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).to(torch.uint8)   # (N,4)
        blk = torch.cat([qs.reshape(-1, 8), e.to(torch.uint8).reshape(-1, 2)], 1)                     # 10 bytes
    elif fmt == "FP3":
        e = group_scale(a, (0.0, 1.0, 2.0, 4.0))
        s = torch.where(UE[e] > 0, UE[e], torch.ones_like(UE[e]))[:, None]
        r = a / s
        mag = torch.where(r == 0, 0, torch.where(r == 1, 1, torch.where(r == 2, 2, 3)))
        code = ((neg & (mag > 0)).to(torch.int32) * 4 | mag).to(torch.int32).reshape(-1, 2, 8)       # (N,2,8)
        c = [code[:, :, i] for i in range(8)]
        b0 = (c[0] & 7) | ((c[1] & 7) << 3) | ((c[2] & 3) << 6)
        b1 = ((c[2] >> 2) & 1) | ((c[3] & 7) << 1) | ((c[4] & 7) << 4) | ((c[5] & 1) << 7)
        b2 = ((c[5] >> 1) & 3) | ((c[6] & 7) << 2) | ((c[7] & 7) << 5)
        qs = torch.stack([b0, b1, b2], 2).reshape(-1, 6).to(torch.uint8)                              # 6 bytes / half
        blk = torch.cat([qs.reshape(-1, 12), e.to(torch.uint8).reshape(-1, 2)], 1)                   # 14 bytes
    else:
        raise SystemExit(f"real packing not implemented for {fmt}")
    return blk.reshape(rows, -1).cpu().numpy()

def quant_block_diag(W, qw, fmt):
    g, dec, _, ql = BLOCK[fmt]
    k = ts_exponent(W, ql)
    x = W * 2.0 ** k
    wts = mse_weights(x, qw)
    rows, cols = x.shape
    xg, wg = x.reshape(-1, g), wts.reshape(-1, g)
    s = choose_scales(xg, wg, dec, BLOCK[fmt][2])
    q = dec(xg, s[:, None]).reshape(rows, cols)
    LAST.update(q=q, k=k)
    return q * 2.0 ** -k

def gptq(W, H, qfun_group, g, blocksize=128, damp=0.01):
    """Generic GPTQ with static groups of g columns. qfun_group(Wcols) -> dequantized cols (rows, g), called at
    each group start on the error-updated weights; columns inside the group are then rounded with the group's
    scales via the returned per-column quantizer."""
    W = W.clone()
    rows, cols = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=W.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    rq = None
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            c = i1 + j
            if c % g == 0:
                rq = qfun_group(W1[:, j:j + g], c)          # returns per-column rounding fn for this group
            w = W1[:, j]
            assert rq is not None
            q = rq(w, c % g)
            Q1[:, j] = q
            e = (w - q) / Hi[j, j]
            W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]
            E1[:, j] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q

def quant_block_gptq(W, H, qw, fmt):
    g, dec, _, ql = BLOCK[fmt]
    k = ts_exponent(W, ql)
    x = W * 2.0 ** k
    sigma2 = (x * x).mean(1, keepdim=True)                   # C++ uses the row energy of the original weights

    def qfun_group(xg, c):
        wts = qw[c:c + g][None, :] * torch.sqrt(sigma2 + xg * xg)
        s = choose_scales(xg, wts, dec, BLOCK[fmt][2])        # one scale per row for this group
        return lambda col, _j: dec(col, s)
    q = gptq(x, H, qfun_group, g)
    LAST.update(q=q, k=k)
    return q * 2.0 ** -k

# ---------------------------------------------------------------- flat row-scaled family (ROCMR)
RLEV = {"R2": 3, "R3": 7}           # levels (2c - L), c in 0..L ; scale s per row
RINIT = {"R2": 0.498, "R3": 0.293}

def rround(x, s, L):
    c = torch.floor((x / s + L) / 2 + 0.5).clamp(0, L)
    return (2 * c - L) * s

def quant_r(W, H, fmt, enc, refit=True):
    L = RLEV[fmt]
    hd = torch.diag(H).clamp_min(1e-12)
    sig = torch.sqrt((W * W * hd[None, :]).sum(1) / hd.sum())          # H-diag-weighted row rms
    s = (RINIT[fmt] * sig).clamp_min(1e-8)
    # coarse 1-D search of the row scale under diag-H weighted error
    best, best_err = s.clone(), torch.full_like(s, float("inf"))
    for f in torch.linspace(0.6, 1.6, 41, device=W.device):
        sf = s * f
        err = (((W - rround(W, sf[:, None], L)) ** 2) * hd[None, :]).sum(1)
        m = err < best_err
        best, best_err = torch.where(m, sf, best), torch.where(m, err, best_err)
    s = best
    if enc == "gptq":
        Q = gptq(W, H, lambda xg, c: (lambda col, _j: rround(col, s, L)), g=W.shape[1])
    else:
        Q = rround(W, s[:, None], L)
    if refit:                                                   # codes frozen, closed-form row scale under H
        C = Q / s[:, None]
        num = ((C @ H) * W).sum(1)
        den = ((C @ H) * C).sum(1).clamp_min(1e-20)
        s2 = torch.where(den > 0, num / den, s)
        Q = C * s2[:, None]
    return Q.half().float()                                     # row scale stored as F16

# ---------------------------------------------------------------- helpers
def hadamard(b, device):
    h = torch.tensor([[(-1) ** bin(i & j).count("1") for j in range(b)] for i in range(b)], dtype=torch.float32)
    return (h / math.sqrt(b)).to(device)

HF2GG = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
         "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}

def load_imatrix(path, gguf):
    r = gguf.GGUFReader(path)
    t = {x.name: np.asarray(x.data, dtype=np.float32) for x in r.tensors}
    out = {}
    for n in t:
        if n.endswith(".in_sum2"):
            base = n[: -len(".in_sum2")]
            cnt = t.get(base + ".counts")
            v = t[n].reshape(-1)
            c = float(np.asarray(cnt).reshape(-1)[0]) if cnt is not None else 1.0
            out[base] = v / max(c, 1.0)
    return out

def main():
    ap = argparse.ArgumentParser()
    for a in ["gguf-py", "hf", "f16", "template", "imatrix", "calib", "out"]:
        ap.add_argument("--" + a, required=True)
    ap.add_argument("--fmt", required=True, choices=["S40", "FP3", "I4", "R2", "R3"])
    ap.add_argument("--enc", default="diag", choices=["diag", "gptq"])
    ap.add_argument("--transform", default="none", choices=["none", "macaq"])
    ap.add_argument("--rot")
    ap.add_argument("--nchunks", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--qprefix", type=int, default=1)
    ap.add_argument("--selftest", type=int, default=0, help="compare diag sim vs a C++ expansion GGUF")
    ap.add_argument("--cpp")
    ap.add_argument("--real", help="C++ real GGUF of the same type as template; writes <out>.real.gguf")
    ap.add_argument("--search", default="cpp", choices=["cpp", "exhaustive"])
    args = ap.parse_args()
    sys.path.insert(0, args.gguf_py)
    import gguf

    global UE, SEARCH
    SEARCH = args.search
    dev = torch.device("cuda")
    UE = ue4m3_table().to(dev)
    t0 = time.time()
    rd = gguf.GGUFReader(args.rot if args.transform == "macaq" else args.f16)
    src = {t.name: t for t in rd.tensors}
    hb = 0
    if args.transform == "macaq":
        hb = int(rd.fields["macaq.hadamard_block"].contents())
        Hb = hadamard(hb, dev)
    im = load_imatrix(args.imatrix, gguf)

    def gg_weight(name):
        t = src[name]
        return torch.from_numpy(np.asarray(t.data).astype(np.float32)).to(dev).reshape(tuple(reversed([int(x) for x in t.shape])))

    if args.selftest:
        cpp = {t.name: t for t in gguf.GGUFReader(args.cpp).tensors}
        rep = []
        for n in ["blk.0.attn_q.weight", "blk.0.ffn_down.weight", "blk.17.ffn_up.weight", "blk.35.attn_output.weight"]:
            W = gg_weight(n)
            q = quant_block_diag(W, torch.from_numpy(im[n]).to(dev), args.fmt)
            c = torch.from_numpy(np.asarray(cpp[n].data).astype(np.float32)).to(dev).reshape(W.shape)
            ts = cpp.get(n[:-7] + ".scale")
            if ts is not None:
                c = c * float(np.asarray(ts.data).reshape(-1)[0])
            exact = (q.half() == c.half()).float().mean().item()
            rel = ((q - c).norm() / c.norm()).item()
            rep.append((n, exact, rel))
            print(f"SELFTEST {n} exact={exact:.6f} relerr={rel:.3e}", flush=True)
        return

    # ---- calibration tokens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = cast(Any, AutoTokenizer.from_pretrained(args.hf))
    ids = tok(Path(args.calib).read_text(encoding="utf-8", errors="ignore"), return_tensors="pt").input_ids[0]
    n = min(args.nchunks, ids.numel() // args.ctx)
    calib = ids[: n * args.ctx].reshape(n, args.ctx)
    assert n > 0, 'no calibration chunks (tokenizer or calib file problem)'
    print(f"calib chunks {n} x {args.ctx}", flush=True)

    need_h = args.enc == "gptq" or args.fmt in ("R2", "R3")
    results, reals = {}, {}
    assert not (args.real and args.transform != 'none'), 'real files only without online transform'
    if need_h:
        model = cast(Any, AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16)).to(dev).eval()
        layers = model.model.layers
        # catch layer-0 inputs
        inps, kw = [], {}

        class Stop(Exception):
            pass

        def catch(mod, a, k):
            inps.append(a[0] if a else k["hidden_states"])
            if not kw:
                kw.update({kk: vv for kk, vv in k.items() if kk != "hidden_states"})
            raise Stop

        h = layers[0].register_forward_pre_hook(catch, with_kwargs=True)
        bs = 8
        with torch.no_grad():
            for i in range(0, n, bs):
                try:
                    model(calib[i:i + bs].to(dev), use_cache=False)
                except Stop:
                    pass
        h.remove()
        print(f"captured {len(inps)} batches, kwargs {sorted(kw)}", flush=True)

        def run_layer(layer, x):
            o = layer(x, **kw)
            return o[0] if isinstance(o, tuple) else o

        for li, layer in enumerate(layers):
            Hs, cnt, hooks = {}, {}, []
            for hfn, ggn in HF2GG.items():
                mod = layer.get_submodule(hfn)
                name = f"blk.{li}.{ggn}.weight"

                def hook(m, a, name=name):
                    x = a[0].reshape(-1, a[0].shape[-1]).float()
                    if name not in Hs:
                        Hs[name] = torch.zeros(x.shape[1], x.shape[1], device=dev)
                        cnt[name] = 0
                    Hs[name].addmm_(x.t(), x)
                    cnt[name] += x.shape[0]
                hooks.append(mod.register_forward_pre_hook(hook))
            with torch.no_grad():
                for x in inps:
                    run_layer(layer, x)
            for hk in hooks:
                hk.remove()
            for hfn, ggn in HF2GG.items():
                name = f"blk.{li}.{ggn}.weight"
                H = Hs.pop(name) / cnt[name]
                W = gg_weight(name)
                if hb:
                    p = torch.from_numpy(np.asarray(src[name[:-7] + ".pre_scale"].data).astype(np.float32).reshape(-1)).to(dev)
                    k = H.shape[0]
                    T = torch.block_diag(*([Hb] * (k // hb)))
                    T = p[:, None] * T                     # x' = (x * p) H  ->  x' = x (D H)
                    H = T.t() @ H @ T
                qw = torch.from_numpy(im[name]).to(dev)
                if args.fmt in ("R2", "R3"):
                    Q = quant_r(W, H, args.fmt, args.enc)
                elif args.enc == "gptq":
                    Q = quant_block_gptq(W, H, qw, args.fmt)
                else:
                    Q = quant_block_diag(W, qw, args.fmt)
                results[name] = Q
                if args.real:
                    reals[name] = (pack_real(args.fmt, LAST['q']), LAST['k'])
                if args.qprefix:
                    Wq = Q
                    if hb:
                        Wq = (Q.reshape(Q.shape[0], -1, hb) @ Hb).reshape(Q.shape) * p[None, :]
                    layer.get_submodule(hfn).weight.data.copy_(Wq.to(torch.bfloat16))
            with torch.no_grad():
                inps = [run_layer(layer, x) for x in inps]
            print(f"layer {li} done {time.time() - t0:.0f}s", flush=True)
        del model
    else:
        for name in src:
            if name.startswith("blk.") and any(name.endswith(f".{g}.weight") for g in HF2GG.values()):
                results[name] = quant_block_diag(gg_weight(name), torch.from_numpy(im[name]).to(dev), args.fmt)
                if args.real:
                    reals[name] = (pack_real(args.fmt, LAST['q']), LAST['k'])

    # ---- export: template (F16 + Q6_K-expanded embd/output), replace linears (un-transformed)
    tr = gguf.GGUFReader(args.template)
    w = gguf.GGUFWriter(args.out, tr.fields["general.architecture"].contents(), endianess=tr.endianess)
    for key, field in tr.fields.items():
        if key.startswith(("GGUF.", "split.")) or key in ("general.architecture", "macaq.hadamard_block"):
            continue
        sub = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(key, field.contents(), field.types[0], sub_type=sub)
    tens = [t for t in tr.tensors if not t.name.endswith((".pre_scale", ".scale")) or t.name in results]
    for t in tens:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.n_bytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in tens:
        if t.name in results:
            Q = results[t.name]
            if hb:
                p = torch.from_numpy(np.asarray(src[t.name[:-7] + ".pre_scale"].data).astype(np.float32).reshape(-1)).to(dev)
                Q = (Q.reshape(Q.shape[0], -1, hb) @ Hb).reshape(Q.shape) * p[None, :]
            a = Q.cpu().numpy().astype(np.float16).reshape(t.data.shape)
            assert t.tensor_type == gguf.GGMLQuantizationType.F16, t.name
            w.write_tensor_data(a)
        else:
            w.write_tensor_data(t.data, tensor_endianess=tr.endianess)
    w.close()
    if args.real:
        rr = gguf.GGUFReader(args.real)
        w = gguf.GGUFWriter(args.out + ".real.gguf", rr.fields["general.architecture"].contents(), endianess=rr.endianess)
        for key, field in rr.fields.items():
            if key.startswith(("GGUF.", "split.")) or key == "general.architecture":
                continue
            sub = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(key, field.contents(), field.types[0], sub_type=sub)
        for t in rr.tensors:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.n_bytes, t.tensor_type)
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
        nrep = nts = 0
        for t in rr.tensors:
            base = t.name[:-6] if t.name.endswith(".scale") else None
            if t.name in reals:
                b = reals[t.name][0]
                assert b.nbytes == t.n_bytes, (t.name, b.nbytes, t.n_bytes)
                w.write_tensor_data(b.reshape(t.data.shape)); nrep += 1
            elif base and base + ".weight" in reals:
                v = np.full(t.data.shape, 2.0 ** -reals[base + ".weight"][1], dtype=np.float32)
                if not np.array_equal(v, np.asarray(t.data, dtype=np.float32)):
                    print(f"TS differs {t.name}: ours {v.reshape(-1)[0]} cpp {np.asarray(t.data).reshape(-1)[0]}", flush=True)
                w.write_tensor_data(v); nts += 1
            else:
                w.write_tensor_data(t.data, tensor_endianess=rr.endianess)
        w.close()
        print(f"REAL written: {nrep} packed tensors, {nts} tensor scales -> {args.out}.real.gguf", flush=True)
    print(f"RQ_DONE {args.fmt} {args.enc} {args.transform} {len(results)} tensors {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
