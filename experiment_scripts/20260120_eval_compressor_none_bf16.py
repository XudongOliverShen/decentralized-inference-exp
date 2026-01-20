"""
Experiment script.

Note: this file lives under `experiment_scripts/`. To make it runnable from any working directory,
we add the repo root to `sys.path` so imports like `from eval_ppl import run_ppl_eval` work.
"""

import gc
import sys
from pathlib import Path

import torch

# Ensure repo-root imports work no matter where you run this from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compressor import Compressor, Payload  # noqa: E402
from eval_ppl import run_ppl_eval  # noqa: E402


class NoneCompressor(Compressor):
    name = "none"
    def compress(self, x: torch.Tensor) -> Payload:
        return Payload(x, {}, x.numel() * x.element_size())
    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return p.data.to(device=device)

for k in [10000, 20000, 50000, 100000, 0]:
    run_ppl_eval(
        wandb_run_name = f"NoneCompressor_bf16_{k}",
        dtype = "bf16",
        load_in_8bit = True,
        compressor=NoneCompressor(),
        first_k_tokens=k,
        result_dir = str(REPO_ROOT / "results" / "compressor_none_bf16"),
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()