"""Turn an F16 expansion of a (possibly MACaQ-transformed, possibly tensor-scaled) quant back into a
plain F16 GGUF of the ORIGINAL model, so stock llama-perplexity can score the weight error.

  w = expansion * ts                    (ts = <name>.scale if present, ROCmFPX tensor scale)
  if <name>.pre_scale exists:  w = (w reshaped (out, in/b, b) @ H_b) * p     (H symmetric orthonormal)

Usage: python -I macaq_unrot.py <gguf-py dir> <expansion.gguf> <out.gguf>
"""
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
import gguf  # noqa: E402


def hadamard(b):
    h = np.array([[(-1) ** bin(i & j).count("1") for j in range(b)] for i in range(b)], dtype=np.float64)
    return (h / np.sqrt(b)).astype(np.float32)


def main():
    src, dst = Path(sys.argv[2]), Path(sys.argv[3])
    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    hb = r.fields.get("macaq.hadamard_block")
    hb = int(hb.contents()) if hb is not None else 0
    H = hadamard(hb) if hb else None
    tens = {t.name: t for t in r.tensors}
    skip = {n for n in tens if n.endswith(".pre_scale") or (n.endswith(".scale") and n[:-6] + ".weight" in tens)}
    w = gguf.GGUFWriter(dst, arch, endianess=r.endianess)
    for key, field in r.fields.items():
        if key.startswith(("GGUF.", "split.")) or key in ("general.architecture", "macaq.hadamard_block"):
            continue
        sub = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(key, field.contents(), field.types[0], sub_type=sub)
    plan, n_rot, n_ts = [], 0, 0
    for t in r.tensors:
        if t.name in skip:
            continue
        base = t.name[: -len(".weight")] if t.name.endswith(".weight") else None
        ps = tens.get(base + ".pre_scale") if base else None
        ts = tens.get(base + ".scale") if base else None
        if ps is None and ts is None:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.n_bytes, t.tensor_type)
            plan.append((t, None, None))
            continue
        shape = tuple(reversed([int(x) for x in t.shape]))  # (out, in)
        w.add_tensor_info(t.name, shape, np.dtype("float16"), int(np.prod(shape)) * 2, gguf.GGMLQuantizationType.F16)
        plan.append((t, ps, ts))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t, ps, ts in plan:
        if ps is None and ts is None:
            w.write_tensor_data(t.data, tensor_endianess=r.endianess)
            continue
        m = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
        oshape = m.shape  # (out, in) or (n_expert, out, in) for merged expert tensors
        n_in = oshape[-1]
        if ts is not None:
            s = np.asarray(ts.data, dtype=np.float32).reshape(-1)
            if s.size == 1:
                m = m * s[0]
            elif m.ndim == 3 and s.size == oshape[0]:          # one scale per expert
                m = m * s.reshape(-1, 1, 1)
            elif s.size * n_in == m.size:                       # one scale per row
                m = m * s.reshape(oshape[:-1] + (1,))
            else:
                raise SystemExit(f"scale shape {s.size} does not fit {t.name} {oshape}")
            n_ts += 1
        m = m.reshape(-1, n_in)
        out = m.shape[0]
        if ps is not None:
            p = np.asarray(ps.data, dtype=np.float32).reshape(-1)
            assert p.size == n_in and n_in % hb == 0, t.name
            assert H is not None, "rotation scales need a Hadamard block size"
            m = (m.reshape(out, n_in // hb, hb) @ H).reshape(out, n_in) * p[None, :]
            n_rot += 1
        if not np.isfinite(m).all():
            raise SystemExit(f"nonfinite {t.name}")
        w.write_tensor_data(m.reshape(oshape).astype(np.float16))
    w.close()
    print(f"UNROT {dst.name}: rotated {n_rot}, tensor-scaled {n_ts}, hadamard_block {hb}")


if __name__ == "__main__":
    main()
