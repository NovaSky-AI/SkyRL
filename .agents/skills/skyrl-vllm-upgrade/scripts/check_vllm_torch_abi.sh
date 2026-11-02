#!/usr/bin/env bash
# Static check: can a vLLM wheel's compiled extensions load against an OLDER torch?
# Use when vLLM pins a newer torch than SkyRL holds via override-dependencies.
#
# Usage: check_vllm_torch_abi.sh <vllm-version> <prev-vllm-version> <held-torch-version> [workdir]
#   e.g. check_vllm_torch_abi.sh 0.33.0 0.32.0 2.13.0
#
# Pass criteria:
#   1. No vllm/**/*.so imports C++ c10::/at::/torch:: symbols (i.e. stable libtorch ABI only).
#   2. Every aoti_torch_*/torch_* shim symbol the new wheel imports is exported by the held
#      torch's libtorch_cpu.so -- except CUDA shim symbols (they live in libtorch_cuda.so,
#      which the CPU wheel lacks); those are accepted if the previous vLLM wheel, which ran
#      on the held torch, already imported them.
# Works on macOS (Apple nm reads ELF) and Linux. Downloads ~1 GB.
set -euo pipefail

new=$1
prev=$2
torch_ver=$3
work=${4:-$(mktemp -d)}
mkdir -p "$work"
echo "workdir: $work"

wheel_url() {  # <version> -> x86_64 manylinux wheel URL
    curl -s "https://pypi.org/pypi/vllm/$1/json" | python3 -c \
        "import json,sys; print([u['url'] for u in json.load(sys.stdin)['urls'] if u['filename'].endswith('x86_64.whl')][0])"
}

shim_syms() {  # <unpacked dir> -> sorted undefined shim symbols
    find "$1" -name '*.so' -exec nm -D --undefined-only {} \; 2>/dev/null |
        awk '{print $2}' | grep -E '^(aoti_torch_|torch_)' | sort -u
}

for v in "$new" "$prev"; do
    if [[ ! -d "$work/vllm-$v" ]]; then
        curl -sL -o "$work/vllm-$v.whl" "$(wheel_url "$v")"
        # unzip's `*` also matches `/`, so this covers nested extensions too.
        unzip -q "$work/vllm-$v.whl" 'vllm/*.so' -d "$work/vllm-$v"
    fi
done

status=0
echo "== 1. non-stable C++ torch symbols per extension ($new)"
while IFS= read -r so; do
    n=$(nm -D --undefined-only "$so" 2>/dev/null | grep -cE '_ZN(K)?(3c10|2at|5torch)' || true)
    printf '%5d  %s\n' "$n" "${so#"$work/vllm-$new/"}"
    [[ "$n" -eq 0 ]] || status=1
done < <(find "$work/vllm-$new" -name '*.so' | sort)

tdir=$work/torch-$torch_ver
if [[ ! -f "$tdir/torch/lib/libtorch_cpu.so" ]]; then
    mkdir -p "$tdir"
    curl -sL -o "$tdir/torch.whl" \
        "https://download.pytorch.org/whl/cpu/torch-${torch_ver}%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"
    unzip -q -o "$tdir/torch.whl" 'torch/lib/libtorch_cpu.so' -d "$tdir"
fi
nm -D --defined-only "$tdir/torch/lib/libtorch_cpu.so" | awk '{print $3}' |
    grep -E '^(aoti_torch_|torch_)' | sort -u > "$work/torch_exports.txt"
shim_syms "$work/vllm-$new" > "$work/shim_new.txt"
shim_syms "$work/vllm-$prev" > "$work/shim_prev.txt"

echo "== 2. shim symbols $new needs that torch $torch_ver's libtorch_cpu.so does not export"
missing=$(comm -23 "$work/shim_new.txt" "$work/torch_exports.txt")
for sym in $missing; do
    if grep -qx "$sym" "$work/shim_prev.txt"; then
        echo "  ok (already used by $prev; CUDA shim): $sym"
    else
        echo "  MISSING: $sym"
        status=1
    fi
done
echo "== new shim symbols vs $prev:"
comm -13 "$work/shim_prev.txt" "$work/shim_new.txt" | sed 's/^/  /'
[[ $status -eq 0 ]] && echo "PASS" || echo "FAIL"
exit $status
