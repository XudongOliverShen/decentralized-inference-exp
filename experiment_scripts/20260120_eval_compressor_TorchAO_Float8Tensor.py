"""
Experiment script.

Note: this file lives under `experiment_scripts/`. To make it runnable from any working directory,
we add the repo root to `sys.path` so imports like `from eval_ppl import run_ppl_eval` work.
"""

import gc
import sys
from pathlib import Path

import torch

from torchao.quantization.quantize_.workflows import Float8Tensor

# Ensure repo-root imports work no matter where you run this from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compressor import Compressor, Payload  # noqa: E402
from eval_ppl import run_ppl_eval  # noqa: E402

def get_torchAOTensor_nbytes(t):
    """
    only tensors are 
    """
    total = 0
    for v in t.__dict__.values():
        if isinstance(v, torch.Tensor):
            total += v.element_size() * v.numel()
    return total


class TorchAO_Float8Tensor_Compressor(Compressor):
    name = "none"
    def compress(self, x: torch.Tensor) -> Payload:
        x_lp = Float8Tensor.from_hp(x)
        return Payload(x_lp, {}, get_torchAOTensor_nbytes(x_lp))
    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return p.data.dequantize().to(device = device, dtype = dtype)

for k in [10000, 20000, 50000, 100000, 0]:
    run_ppl_eval(
        wandb_run_name = f"TorchAO_Float8Tensor_Compressor_{k}",
        dtype = "bf16",
        load_in_8bit = True,
        compressor=TorchAO_Float8Tensor_Compressor(),
        first_k_tokens=k,
        result_dir = str(REPO_ROOT / "results" / "compressor_TorchAO_Float8Tensor"),
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()