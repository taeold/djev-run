"""Copy the fork's changed vllm/ files over the base image's vLLM.

usage: overlay_vllm.py <fork checkout> <base commit>

The checkout carries changed.txt, the paths under vllm/ that differ from the
base commit, and added.txt, the subset the fork creates. The base image's vLLM
must report that commit in its version, and a target outside added.txt must
already exist, or the build stops: an overlay onto a different vLLM would
import and then fail in ways that look like model bugs.
"""

import importlib.metadata
import importlib.util
import pathlib
import shutil
import sys

fork = pathlib.Path(sys.argv[1])
base = sys.argv[2]

version = importlib.metadata.version("vllm")
short = base[:9]
if f"+g{short}" not in version:
  raise SystemExit(
      f"base image vLLM is {version}; the overlay expects commit {short}. Bump"
      " VLLM_BASE and VLLM_REF together, against a fork branch built on that"
      " commit."
  )

_spec = importlib.util.find_spec("vllm")
assert _spec is not None and _spec.origin is not None
site = pathlib.Path(_spec.origin).parent


def paths(name):
  f = fork / name
  if not f.exists():
    return []
  return [line.strip() for line in f.read_text().splitlines() if line.strip()]


changed = paths("changed.txt")
added = set(paths("added.txt"))
if not changed:
  raise SystemExit(
      "changed.txt is empty; the fork ref carries no vllm/ changes"
  )

for rel in changed:
  src = fork / rel
  dst = site / pathlib.Path(rel).relative_to("vllm")
  if not dst.exists():
    if rel not in added:
      raise SystemExit(f"{dst} is not in the base image; the base moved")
    dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(src, dst)
  for pyc in (dst.parent / "__pycache__").glob(dst.stem + ".*.pyc"):
    pyc.unlink()
  print("overlay", rel)
print(f"overlaid {len(changed)} files onto vllm {version}")
