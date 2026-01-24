"""
Experiment script.

Note: this file lives under `experiment_scripts/`. To make it runnable from any working directory,
we add the repo root to `sys.path` so imports like `from eval_ppl import run_ppl_eval` work.
"""

import gc
import sys
from pathlib import Path

import torch

from torchao.quantization.quant_primitives import (
    choose_qparams_affine, quantize_affine, dequantize_affine,
    MappingType,
)

# Ensure repo-root imports work no matter where you run this from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compressor import Compressor, Payload  # noqa: E402
from eval_ppl import run_ppl_eval  # noqa: E402

def tensor_nbytes(t: torch.Tensor) -> int:
    return t.element_size() * t.numel()

def packed_nbytes(numel: int, bits: int) -> int:
    # ceil(numel*bits/8)
    return (numel * bits + 7) // 8


def _qrange(bits: int, unsigned: bool):
    assert 2 <= bits <= 8
    if unsigned:
        return 0, (2 ** bits) - 1
    else:
        return -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1

def _make_block_size(x: torch.Tensor, per_token: bool):
    # TorchAO 要求 block_size 长度 == x.dim() :contentReference[oaicite:2]{index=2}
    if not per_token:
        # per-tensor: block_size == input_size
        return tuple(x.shape)

    # per-token/per-row（最后一维共享一个 scale；其他维度各自独立）
    # 等价于：对除了最后一维外，每个 “row” 单独算 qparams
    # block_size = (1,1,..., D)  (len == x.dim())
    return (1,) * (x.dim() - 1) + (x.shape[-1],)

@torch.no_grad()
def affine_intx_quant(
    x: torch.Tensor,
    bits: int = 2,
    *,
    per_token: bool = False,
    mapping_type: MappingType = MappingType.ASYMMETRIC,
    unsigned: bool = True,
):
    """
    Returns:
      q (int8 container), scale, zp (optional tensor), qrange (qmin,qmax), meta dict
    Note:
      q 的真实存储 dtype 仍是 int8（未做 bit-pack），只是数值被 clamp 到 int{bits} 范围。
    """
    assert 2 <= bits <= 8, "bits must be in [2, 8]"
    qmin, qmax = _qrange(bits, unsigned)
    block_size = _make_block_size(x, per_token)
    container_dtype = torch.uint8 if unsigned else torch.int8  # <- 建议配合 qrange

    scale, zp = choose_qparams_affine(
        x,
        mapping_type=mapping_type,
        block_size=block_size,
        target_dtype=container_dtype,
        quant_min=qmin,
        quant_max=qmax,
    )


    q = quantize_affine(
        x,
        block_size=block_size,
        scale=scale,
        zero_point=zp,
        output_dtype=container_dtype,
        quant_min=qmin,
        quant_max=qmax,
    )

    zp = zp.to(torch.int16) # default is int32, but not necessary?

    meta = {
        "shape": tuple(x.shape),
        "per_token": per_token,
        "bits": bits,
        "mapping_type": mapping_type,
        "unsigned": unsigned,
    }
    return q, scale, zp, (qmin, qmax), meta

@torch.no_grad()
def affine_intx_dequant(
    q: torch.Tensor,
    scale: torch.Tensor,
    zp,
    qrange,
    *,
    shape,
    per_token: bool,
    out_dtype: torch.dtype = torch.float16,
):
    qmin, qmax = qrange
    zp = zp.to(torch.int32)
    # shape 用来重建 block_size
    dummy = torch.empty(shape, device=q.device)
    block_size = _make_block_size(dummy, per_token)

    x_hat = dequantize_affine(
        q,
        block_size=block_size,
        scale=scale,
        zero_point=zp,
        input_dtype=q.dtype,
        quant_min=qmin,
        quant_max=qmax,
        output_dtype=out_dtype,
    )
    return x_hat

class TorchAO_IntXAffine_Compressor(Compressor):
    def __init__(self, bits: int = 8, per_token: bool = True, use_theoretical_size: bool = True):
        assert 2 <= bits <= 8
        self.bits = bits
        self.per_token = per_token
        self.use_theoretical_size = use_theoretical_size
        self.name = f"torchao_int{bits}_{'pertoken' if per_token else 'pertensor'}"

    def compress(self, x: torch.Tensor) -> Payload:
        q, scale, zp, qrange, meta = affine_intx_quant(
            x, bits=self.bits, per_token=self.per_token
        )

        aux = {
            "scale": scale,
            "zero_point": zp,
            "qrange": qrange,
            "meta": meta,
        }

        if self.use_theoretical_size:
            total = packed_nbytes(q.numel(), self.bits) + tensor_nbytes(scale)
            if isinstance(zp, torch.Tensor) and zp.numel() > 0:
                total += tensor_nbytes(zp)
        else:
            # 实际存储（q 是 int8 容器）
            total = tensor_nbytes(q) + tensor_nbytes(scale)
            if isinstance(zp, torch.Tensor) and zp.numel() > 0:
                total += tensor_nbytes(zp)

        return Payload(q, aux, total)

    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q = p.data
        scale = p.meta["scale"]
        zp = p.meta["zero_point"]
        qrange = p.meta["qrange"]
        meta = p.meta["meta"]

        # 注意：q/scale/zp 可能在 CPU；搬到目标 device
        q = q.to(device=device)
        scale = scale.to(device=device)
        if isinstance(zp, torch.Tensor) and zp.numel() > 0:
            zp = zp.to(device=device)

        x_hat = affine_intx_dequant(
            q, scale, zp, qrange,
            shape=meta["shape"],
            per_token=meta["per_token"],
            out_dtype=dtype,
        )
        return x_hat

for k in [10000, 0]:
    for bits in [8,2,4,7,6,5,3]:
        run_ppl_eval(
            wandb_run_name = f"TorchAO_IntXAffine_{bits}bit_Compressor_{k}",
            dtype = "bf16",
            load_in_8bit = True,
            compressor=TorchAO_IntXAffine_Compressor(bits=bits),
            first_k_tokens=k,
            result_dir = str(REPO_ROOT / "results" / f"compressor_TorchAO_IntXAffine_{bits}bit"),
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()