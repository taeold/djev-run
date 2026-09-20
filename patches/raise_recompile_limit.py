"""Raise torch dynamo's recompile limit for the DiffusionGemma sampler.

The compiled sample step is one dynamo specialization per canvas width, and
reads through the structured server arrive in several widths. Past torch's
default of 8 recompiles dynamo runs the function eager for the rest of the
process, which is slower and was where an eager-only dtype mismatch surfaced.
There is no environment variable for the limit, so the raise is written into
the module above the sample step's decorator, once.
"""

import importlib.util
import pathlib

_spec = importlib.util.find_spec("vllm")
assert _spec is not None and _spec.origin is not None
site = pathlib.Path(_spec.origin).parent
target = site / "model_executor" / "models" / "diffusion_gemma.py"
DEF = "def _compiled_sample_step(\n"
MARKER = "# [djev-spark] recompile limit"
INSERT = f"""{MARKER}
# Reads come in several canvas widths and each width is a fresh dynamo
# specialization of the sampler. Past torch's default of 8 every later width
# runs eager for the rest of the process.
for _name in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _name):
        setattr(
            torch._dynamo.config,
            _name,
            max(getattr(torch._dynamo.config, _name), 64),
        )


"""

text = target.read_text()
if MARKER in text:
  print("recompile limit already raised")
  raise SystemExit(0)
if text.count(DEF) != 1:
  raise SystemExit(
      f"expected exactly one {DEF.strip()} in {target.name}, found"
      f" {text.count(DEF)}"
  )
lines = text.split("\n")
at = next(i for i, line in enumerate(lines) if line + "\n" == DEF)
if at == 0 or not lines[at - 1].startswith("@torch.compile("):
  raise SystemExit(
      "the sample step is no longer directly under a torch.compile decorator;"
      " re-check the anchor"
  )
lines[at - 1 : at - 1] = INSERT.rstrip("\n").split("\n") + ["", ""]
target.write_text("\n".join(lines))
for pyc in (target.parent / "__pycache__").glob("diffusion_gemma.*.pyc"):
  pyc.unlink()
print("raised the sampler's recompile limit to 64")
