#!/usr/bin/env bash
# Download and extract a user-local ROCm 7.2.1 rpath toolchain.
#
# This intentionally does not modify system apt sources or install packages.
# Every .deb is checked against the SHA256 listed in the ROCm package index before it is extracted.

set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-7.2.1}"
UBUNTU_CODENAME="${UBUNTU_CODENAME:-jammy}"
ROOT="${ROOT:-$HOME/rocm-${ROCM_VERSION}-local}"
BASE_URL="${BASE_URL:-https://repo.radeon.com/rocm/apt/${ROCM_VERSION}}"
PKG_GZ="$ROOT/apt/Packages.gz"
DEB_DIR="$ROOT/debs"
EXTRACT_DIR="$ROOT/extract"
ROCM_PATH_LOCAL="$EXTRACT_DIR/opt/rocm-${ROCM_VERSION}"
COMPAT_LIB_DIR="$ROOT/compat/extract/usr/lib/x86_64-linux-gnu"
RPATH_SUFFIX="rpath${ROCM_VERSION}"

mkdir -p "$ROOT/apt" "$DEB_DIR" "$EXTRACT_DIR"

# https only, fail on HTTP errors instead of saving an error page
fetch() { curl -fL --proto '=https' --tlsv1.2 -o "$2.part" "$1" && mv "$2.part" "$2"; }

if [[ ! -s "$PKG_GZ" ]]; then
    fetch "$BASE_URL/dists/${UBUNTU_CODENAME}/main/binary-amd64/Packages.gz" "$PKG_GZ"
fi

manifest="$ROOT/manifest.tsv"
python3 - "$PKG_GZ" "$manifest" "$RPATH_SUFFIX" <<'PY'
import gzip
import sys

pkg_gz, manifest, rpath_suffix = sys.argv[1], sys.argv[2], sys.argv[3]
text = gzip.open(pkg_gz, "rt", encoding="utf-8", errors="replace").read()
records = {}

for block in text.strip().split("\n\n"):
    data = {}
    key = None
    for line in block.splitlines():
        if line.startswith(" "):
            if key:
                data[key] += " " + line.strip()
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            data[key] = value.strip()
    if "Package" in data:
        records[data["Package"]] = data

start = [
    f"hip-dev-{rpath_suffix}",
    f"rocm-cmake-{rpath_suffix}",
    f"rocm-device-libs-{rpath_suffix}",
    f"rocblas-dev-{rpath_suffix}",
    f"hipblas-dev-{rpath_suffix}",
    f"rocwmma-dev-{rpath_suffix}",
]

seen = []
queue = list(start)

while queue:
    package = queue.pop(0)
    if package in seen:
        continue
    if package not in records:
        raise SystemExit(f"missing package in ROCm index: {package}")
    seen.append(package)

    deps = records[package].get("Depends", "")
    for dep in deps.split(","):
        token = dep.strip()
        if not token:
            continue
        # Prefer the first alternative. System dependencies such as libc6 are
        # intentionally ignored; only ROCm rpath packages are staged locally.
        name = token.split("|")[0].strip().split()[0]
        if name in records and name.endswith(rpath_suffix):
            if name not in seen and name not in queue:
                queue.append(name)

with open(manifest, "w", encoding="utf-8") as fh:
    for package in seen:
        rec = records[package]
        sha = rec.get("SHA256", "")
        if len(sha) != 64:
            raise SystemExit(f"no SHA256 for {package} in the ROCm index")
        fh.write(f"{package}\t{rec['Filename']}\t{rec.get('Size', '0')}\t{sha}\n")

total = sum(int(records[p].get("Size", "0")) for p in seen)
print(f"packages={len(seen)} download_bytes={total}")
PY

while IFS=$'\t' read -r package filename size sha; do
    deb="$DEB_DIR/${filename##*/}"
    if [[ ! -s "$deb" ]]; then
        echo "download $package ($size bytes)"
        fetch "$BASE_URL/$filename" "$deb"
    else
        echo "reuse $package"
    fi
    if ! echo "$sha  $deb" | sha256sum -c --status -; then
        echo "SHA256 mismatch for $deb (expected $sha), removing it" >&2
        rm -f "$deb"
        exit 1
    fi
done < "$manifest"

while IFS=$'\t' read -r package filename size sha; do
    deb="$DEB_DIR/${filename##*/}"
    echo "extract $package"
    dpkg-deb -x "$deb" "$EXTRACT_DIR"
done < "$manifest"

if [[ ! -x "$ROCM_PATH_LOCAL/bin/hipcc" ]]; then
    echo "missing expected hipcc at $ROCM_PATH_LOCAL/bin/hipcc" >&2
    exit 1
fi

cat > "$ROOT/env.sh" <<EOF
export ROCM_PATH="$ROCM_PATH_LOCAL"
export HIP_PATH="$ROCM_PATH_LOCAL"
export HIP_PLATFORM=amd
export PATH="$ROCM_PATH_LOCAL/bin:\$PATH"
export LD_LIBRARY_PATH="$COMPAT_LIB_DIR:$ROCM_PATH_LOCAL/lib:$ROCM_PATH_LOCAL/llvm/lib:$ROCM_PATH_LOCAL/lib/rocprofiler-systems:\${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ROCM_PATH_LOCAL:\${CMAKE_PREFIX_PATH:-}"
EOF

echo "ROCm ${ROCM_VERSION} local path: $ROCM_PATH_LOCAL"
echo "Environment file: $ROOT/env.sh"
"$ROCM_PATH_LOCAL/bin/hipcc" --version
