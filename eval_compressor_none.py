# compressors/fp16_cast.py
import gc
import torch
from compressor import Compressor, Payload

from eval_ppl import run_ppl_eval


class NoneCompressor(Compressor):
    name = "none"
    def compress(self, x: torch.Tensor) -> Payload:
        return Payload(x, {}, x.numel() * x.element_size())
    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return p.data.to(device=device)

for k in [10000, 20000, 50000, 100000, 0]:
    run_ppl_eval(
        wandb_run_name = f"NoneCompressor_{k}",
        load_in_8bit = True,
        compressor=NoneCompressor(),
        first_k_tokens=k,
        result_dir = './results/compressor_none'
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()