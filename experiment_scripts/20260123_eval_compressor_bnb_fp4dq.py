"""
Experiment script.

Note: this file lives under `experiment_scripts/`. To make it runnable from any working directory,
we add the repo root to `sys.path` so imports like `from eval_ppl import run_ppl_eval` work.
"""

import gc
import sys
from pathlib import Path

import torch

import bitsandbytes as bnb

# Ensure repo-root imports work no matter where you run this from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compressor import Compressor, Payload  # noqa: E402
from eval_ppl import run_ppl_eval  # noqa: E402

def get_bnb_nbytes(x_lp, state):
    """
    only tensors are 
    """
    total = x_lp.element_size() * x_lp.numel()
    for v in state.__dict__.values():
        if isinstance(v, torch.Tensor):
            total += v.element_size() * v.numel()
    return total


class bnb_fp4_Compressor(Compressor):
    name = "none"
    def compress(self, x: torch.Tensor) -> Payload:
        x_lp, state = bnb.functional.quantize_4bit(x, quant_type="fp4-dq")  # packed uint8 + state
        return Payload(x_lp, {"state": state}, get_bnb_nbytes(x_lp, state))
    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        x_hp = bnb.functional.dequantize_4bit(p.data, quant_state=p.meta['state']).to(device = device, dtype = dtype)
        return x_hp

for k in [10000, 20000, 50000, 100000, 0]:
    run_ppl_eval(
        wandb_run_name = f"bnb_fp4dq_Compressor_{k}",
        dtype = "bf16",
        load_in_8bit = True,
        compressor=bnb_fp4_Compressor(),
        first_k_tokens=k,
        result_dir = str(REPO_ROOT / "results" / "compressor_bnb_fp4dq"),
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()