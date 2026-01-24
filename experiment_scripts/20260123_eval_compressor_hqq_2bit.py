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

import torch.nn as nn
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear  # HQQ 官方接口 :contentReference[oaicite:2]{index=2}

# Ensure repo-root imports work no matter where you run this from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compressor import Compressor, Payload  # noqa: E402
from eval_ppl import run_ppl_eval  # noqa: E402

def _tensor_nbytes(obj) -> int:
    """递归统计 obj(可能是 tensor / dict / list) 里所有 tensor 的字节数。"""
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.numel()
    if isinstance(obj, dict):
        return sum(_tensor_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_nbytes(v) for v in obj)
    return 0


def _pick_group_size(dim: int, preferred: int) -> int:
    """
    HQQ 的 group_size 要能整除被分组的维度（一般是 weight.shape[axis]）。
    如果 preferred 不能整除，就降到 dim 的一个因子（尽量大，保证压缩率）。
    """
    if dim <= 0:
        return 1
    if preferred is None:
        return dim
    if dim % preferred == 0:
        return preferred
    # 找 <= preferred 的最大因子
    for g in range(min(preferred, dim), 0, -1):
        if dim % g == 0:
            return g
    return 1

class hqq_4bit_Compressor(Compressor):
    name = "hqq"
    def __init__(self, nbits: int = 2, group_size: int = 64, axis: int = 1):
        """
        常用建议：nbits=4, group_size=64, axis=1（HQQ README 也这么建议）:contentReference[oaicite:3]{index=3}
        """
        self.nbits = nbits
        self.group_size = group_size
        self.axis = axis

    def compress(self, x: torch.Tensor) -> Payload:
        # 1) reshape -> 2D (把最后一维当作 in_features，其余维度合并成 out_features)
        orig_shape = tuple(x.shape)
        if x.ndim == 1:
            x2d = x.view(1, -1)
        else:
            x2d = x.reshape(-1, x.shape[-1])

        # 2) 选一个合法 group_size（要整除被分组的维度）
        dim_for_group = x2d.shape[self.axis]
        gs = _pick_group_size(dim_for_group, self.group_size)

        quant_cfg = BaseQuantizeConfig(nbits=self.nbits, group_size=gs, axis=self.axis)

        # 3) 构造 dummy Linear，并把 weight 塞成 x2d
        #    线性层 weight 形状是 [out_features, in_features]，正好对应 x2d
        linear = nn.Linear(x2d.shape[1], x2d.shape[0], bias=False).to(device=x.device, dtype=x.dtype)
        with torch.no_grad():
            linear.weight.copy_(x2d)

        # 4) 用 HQQLinear 量化（初始化时就会把 weight 变成 W_q + meta）
        #    compute_dtype 用来控制 dequantize/计算时的 dtype（可按你的评测 dtype 改）
        hqq_layer = HQQLinear(
            linear,
            quant_config=quant_cfg,
            compute_dtype=x.dtype,
            device=x.device,
            del_orig=True,
        )

        # 5) 保存完整 state_dict（最稳），同时把 W_q 作为 payload.data（主数据）
        #    不同版本 HQQ 的 key 可能略有变化，所以不要只存 scale/zero。
        st = hqq_layer.state_dict()

        if "W_q" not in st:
            raise KeyError(f"HQQLinear.state_dict() 里找不到 'W_q'，实际 keys={list(st.keys())[:20]}...")

        data = st["W_q"]
        meta = {
            "hqq_state_dict": {k: v for k, v in st.items() if k != "W_q"},
            "orig_shape": orig_shape,
            "reshaped_2d": tuple(x2d.shape),
            "compute_dtype": str(x.dtype).replace("torch.", ""),
            "nbits": self.nbits,
            "group_size": gs,
            "axis": self.axis,
        }

        nbytes = _tensor_nbytes(data) + _tensor_nbytes(meta["hqq_state_dict"])
        return Payload(data, meta, nbytes)
    
    
    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # 1) 重建 state_dict（把 W_q 放回去）
        st = dict(p.meta["hqq_state_dict"])
        st["W_q"] = p.data

        # 2) 用“空壳” HQQLinear + load_state_dict 恢复（Transformers 的做法也是这样）:contentReference[oaicite:4]{index=4}
        hqq_layer = HQQLinear(
            linear_layer=None,
            quant_config=None,
            compute_dtype=dtype,
            device=device,
            del_orig=False,
        )
        hqq_layer.load_state_dict(st)

        # 3) 反量化回 2D weight，再 reshape 回原形状
        x2d = hqq_layer.dequantize()  # shape: [out_features, in_features]
        orig_shape = tuple(p.meta["orig_shape"])
        if len(orig_shape) == 1:
            x = x2d.view(-1)
        else:
            x = x2d.reshape(orig_shape)

        return x.to(device=device, dtype=dtype)

# for k in [10000]:
for k in [10000, 20000, 50000, 100000, 0]:
    run_ppl_eval(
        wandb_run_name = f"hqq_2bit_Compressor_{k}",
        dtype = "bf16",
        load_in_8bit = True,
        compressor=hqq_4bit_Compressor(),
        first_k_tokens=k,
        result_dir = str(REPO_ROOT / "results" / "compressor_hqq_2bit"),
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()