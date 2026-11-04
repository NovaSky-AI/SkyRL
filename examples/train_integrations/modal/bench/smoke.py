"""B300 smoke test: SkyRL's megatron-extra stack imports, a bf16 matmul, and a tiny vLLM generation."""

import time

import torch

t = time.time()
import megatron.core  # noqa: E402,F401
import transformer_engine  # noqa: E402,F401
import vllm  # noqa: E402

print(f"SMOKE imports ok in {time.time() - t:.0f}s: vllm {vllm.__version__}, torch {torch.__version__}")
print(f"SMOKE device {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)}")
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    a @ a
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(20):
    a @ a
torch.cuda.synchronize()
print(f"SMOKE bf16 matmul {20 * 2 * 8192**3 / (time.perf_counter() - t) / 1e12:.0f} TFLOP/s")

if __name__ == "__main__":
    from vllm import LLM, SamplingParams  # noqa: E402
    
    llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.3, max_model_len=2048)
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16, temperature=0))
    print(f"SMOKE vllm generate ok: {out[0].outputs[0].text!r}")
