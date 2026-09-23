"""Apply upstream vLLM block-diffusion PR #57250 overlay and eager startup patch."""

import importlib.util
import pathlib
import shutil
import urllib.request

VLLM_57250 = "f831226339b87554b1de9341276e7bee7270fa8a"
FILES_57250 = (
    "model_executor/models/diffusion_gemma.py",
    "utils/diffusion.py",
    "v1/core/sched/diffusion_scheduler.py",
    "config/diffusion.py",
    "v1/worker/gpu/model_runner.py",
    "v1/worker/gpu/states.py",
)


def patch_vllm(vllm_dir: pathlib.Path | None = None) -> None:
    if vllm_dir is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("Could not locate installed vllm package")
        vllm_dir = pathlib.Path(spec.origin).resolve().parent

    print(f"[patch_vllm] Applying vLLM block-diffusion PR #57250 to {vllm_dir}...")
    for rel in FILES_57250:
        dest = vllm_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/mmastrac/vllm/{VLLM_57250}/vllm/{rel}",
            dest,
        )

    baked_vllm = pathlib.Path("/opt/dgemma/v9_baked/vllm")
    if baked_vllm.parent.exists():
        for rel in FILES_57250:
            dest = baked_vllm / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(vllm_dir / rel, dest)

    p = vllm_dir / "v1/worker/gpu_worker.py"
    if p.exists():
        s = p.read_text()
        if "while os.path.exists('/dev/shm/dgemma')" not in s:
            s = s.replace(
                "        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:\n"
                "            # still need a profile run which compiles the model for\n"
                "            # max_num_batched_tokens\n"
                "            self.model_runner.profile_run()",
                "        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:\n"
                "            if not self.model_config.enforce_eager:\n"
                "                self.model_runner.profile_run()",
            )
            s = s.replace(
                "        kernel_warmup(self)\n\n"
                "        if self.use_v2_model_runner:\n"
                "            # A workspace resize after capture frees what the graphs point at.\n"
                "            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)",
                "        if not self.model_config.enforce_eager:\n"
                "            kernel_warmup(self)\n"
                "            if self.use_v2_model_runner:\n"
                "                warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)",
            )
            s = s.replace(
                "self.model_runner.load_model(load_dummy_weights=load_dummy_weights)",
                "import os, time\n"
                "            while os.path.exists('/dev/shm/dgemma') and not os.path.exists('/dev/shm/dgemma/.ready'): time.sleep(0.02)\n"
                "            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)",
            )
            p.write_text(s)
    print("[patch_vllm] vLLM block-diffusion PR #57250 applied.")


if __name__ == "__main__":
    patch_vllm()
