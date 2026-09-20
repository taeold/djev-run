#!/usr/bin/env bash
# Put the CUDA headers where nvcc looks for them.
#
# This base ships a trimmed /usr/local/cuda/include: the libraries are all
# present (libnvrtc.so.13 included) but the matching headers are not. The pip
# nvidia-cu13 package carries the full set, so the fix is to link what is
# missing rather than install anything.
#
# The same trimming applies to the development symlinks: libnvrtc.so.13 is
# present but libnvrtc.so is not, and -lnvrtc resolves against the unversioned
# name. (libcudart.so is there, which is why only some links fail.)
#
# Without the headers, FlashInfer's DeepGEMM JIT dies compiling the CUTLASS MoE
# kernels at step 17 of 97:
#
#   jit_utils.cuh:21:10: fatal error: nvrtc.h: No such file or directory
#
# and without the symlink it gets to 78 of 78 and dies linking:
#
#   /usr/bin/ld: cannot find -lnvrtc
#
# Either way the worker takes SIGKILL and the engine never starts, which reads
# as a hang rather than a build failure.
set -euo pipefail

SRC=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
DST=/usr/local/cuda/include
[[ -d "$SRC" ]] || { echo "no $SRC; base image changed" >&2; exit 1; }

linked=0
for h in "$SRC"/*; do
  name=$(basename "$h")
  [[ -e "$DST/$name" ]] && continue
  ln -s "$h" "$DST/$name"
  linked=$(( linked + 1 ))
done

# The header that motivated this must now resolve on the default include path.
echo '#include <nvrtc.h>' > /tmp/_nvrtc_probe.c
gcc -fsyntax-only -I"$DST" /tmp/_nvrtc_probe.c
rm -f /tmp/_nvrtc_probe.c
# -l<name> needs the unversioned symlink, which this base ships for some
# libraries and not others.
LIB=/usr/local/cuda/lib64
solinks=0
for so in "$LIB"/lib*.so.[0-9]*; do
  base=$(basename "$so"); stem=${base%%.so.*}
  [[ -e "$LIB/$stem.so" ]] && continue
  ln -s "$so" "$LIB/$stem.so"
  solinks=$(( solinks + 1 ))
done
ldconfig 2>/dev/null || true
[[ -e "$LIB/libnvrtc.so" ]] || { echo "libnvrtc.so still missing" >&2; exit 1; }

echo "[cuda-headers] linked $linked headers and $solinks dev symlinks; nvrtc resolves"
