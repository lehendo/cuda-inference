#!/bin/bash
# Static evidence that the Tensor Core backend really issues tensor-core MMA instructions:
# counts HMMA/IMMA SASS instructions per kernel in a built binary, with the SIMT kernels as the
# negative control (they must contain none).
set -euo pipefail
source "$(dirname "$0")/env.sh"
BIN="${1:-$GPU_BUILD_DIR/tests/test_tensor_core}"
cuobjdump -sass "$BIN" > "${TMPDIR:-/tmp}/gie_sass_$$.txt"
python3 - "${TMPDIR:-/tmp}/gie_sass_$$.txt" <<'PY'
import re, sys, collections
fn=None; counts=collections.defaultdict(int); mma=collections.defaultdict(int)
for line in open(sys.argv[1], errors="ignore"):
    m = re.match(r"\s*Function : (\S+)", line)
    if m: fn=m.group(1); counts[fn]+=0; continue
    if fn and re.search(r"\b(HMMA|IMMA|DMMA|BMMA)\.", line): mma[fn]+=1
    if fn and re.search(r"/\*[0-9a-f]{4}\*/", line): counts[fn]+=1
groups = {"tensor_core": [], "simt (implicit_gemm/simt_tiled/direct)": [], "other": []}
for f in counts:
    key = "tensor_core" if ("tc_conv_kernel" in f or "tcmma_kernel" in f) else ("simt (implicit_gemm/simt_tiled/direct)" if any(s in f for s in ("implicit_gemm_kernel","simt_tiled_kernel","direct_conv_kernel")) else "other")
    groups[key].append(f)
ok=True
for g,fs in groups.items():
    if not fs: continue
    tot=sum(mma[f] for f in fs)
    print(f"{g}: {len(fs)} kernels, {tot} MMA instructions, min/kernel {min(mma[f] for f in fs)}")
    if g=="tensor_core" and any(mma[f]==0 for f in fs): ok=False
    if g.startswith("simt") and tot!=0: ok=False
print("RESULT:", "PASS (tensor kernels contain HMMA; SIMT kernels contain none)" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
