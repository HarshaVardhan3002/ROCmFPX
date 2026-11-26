"""Check the RIQ packer against llama-quantize: same tensor, plain encoder (H=None), compare packed bytes with the C
file, and check that our F16 values equal the C dequant of OUR packed bytes (decoded here like rocmfpx_rocmiq*_unpack).
usage: riq_check.py GGUF_PY F16 IMATRIX CFILE HDR FMT"""
import sys
import numpy as np
import torch

gp, f16, imf, cfile, hdr, fmt = sys.argv[1:7]
sys.path.insert(0, gp)
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import gguf
import rq3

dev = torch.device("cuda")
rq3.UE = rq3.ue4m3_table().to(dev)
rq3.setup_vq(dev, hdr)
im = rq3.load_imatrix(imf, gguf)
src = {t.name: t for t in gguf.GGUFReader(f16).tensors}
cpp = {t.name: t for t in gguf.GGUFReader(cfile).tensors}
UE = rq3.UE.cpu().numpy()
KF = np.array([1.0, 0.70710678118654752, 0.5, 0.35355339059327376], dtype=np.float32)
grid = None
if fmt == "RIQ2":
    g = rq3.iq2xxs_grid(hdr, dev).cpu().numpy().astype(np.int32)            # (256, 8) bytes
cb = np.round(rq3.VQ["RIQ1"][1].cpu().numpy() * 32).astype(np.int32) if fmt == "RIQ1" else None

def decode(b, rows, cols):
    nb = cols // 256
    if fmt == "RIQ2":
        w = b.reshape(rows, nb, 33, 2).astype(np.int32)
        w = w[..., 0] | (w[..., 1] << 8)
        scw, codes = w[:, :, 0], w[:, :, 1:]
        signs = (codes >> 8) & 0x7F
        par = signs.copy()
        for sh in (4, 2, 1):
            par ^= par >> sh
        signs |= (par & 1) << 7
        mag = g[codes & 0xFF]                                                   # (rows, nb, 32, 8)
        sg = ((signs[..., None] >> np.arange(8)) & 1).astype(bool)
        q = np.where(sg, -mag, mag) + np.where((codes >> 15)[..., None] == 1, -4, 4)
        q = q.reshape(rows, nb, 4, 64)
        div = 0.125
    else:
        bb = b.reshape(rows, nb, 50).astype(np.int32)
        scw = bb[:, :, 0] | (bb[:, :, 1] << 8)
        ql, qh = bb[:, :, 2:34], bb[:, :, 34:50]
        i = np.arange(64)
        ci = ((ql[:, :, i // 2] >> (4 * (i % 2))) & 15) | (((qh[:, :, i // 4] >> (2 * (i % 4))) & 3) << 4)
        assert cb is not None
        q = cb[ci].reshape(rows, nb, 4, 64)
        div = 0.03125
    sub = UE[scw & 0xFF][..., None] * KF[(scw[..., None] >> (8 + 2 * np.arange(4))) & 3]   # (rows, nb, 4)
    return (q * (sub * div)[..., None].astype(np.float32)).reshape(rows, cols)

for n in ["blk.0.attn_q.weight", "blk.17.ffn_up.weight", "blk.35.ffn_down.weight"]:
    t = src[n]
    W = torch.from_numpy(np.asarray(t.data).astype(np.float32)).to(dev).reshape(tuple(reversed([int(x) for x in t.shape])))
    rows, cols = W.shape
    Q = rq3.quant_vq(W, None, torch.from_numpy(im[n]).to(dev), fmt)
    ours, k = rq3.LAST.pop("blk"), rq3.LAST["k"]
    c = np.asarray(cpp[n].data).view(np.uint8).reshape(rows, -1)
    ts = cpp.get(n[:-7] + ".scale")
    cts = float(np.asarray(ts.data).reshape(-1)[0]) if ts is not None else 1.0
    same_bytes = (ours == c).mean()
    per = 66 if fmt == "RIQ2" else 50
    same_blocks = (ours.reshape(-1, per) == c.reshape(-1, per)).all(1).mean()
    dq = decode(ours, rows, cols) * 2.0 ** -k
    exact = np.abs(dq - Q.cpu().numpy()).max()
    dc = decode(c, rows, cols) * cts
    w = W.cpu().numpy()
    e_ours = np.square(dq - w).sum() / np.square(w).sum()
    e_c = np.square(dc - w).sum() / np.square(w).sum()
    print(f"{n} TS ours 2^-{k} cpp {cts:g} | same bytes {same_bytes:.4f} blocks {same_blocks:.4f} | "
          f"values==decode max diff {exact:.2e} | rel err ours {e_ours:.5f} cpp {e_c:.5f}", flush=True)
