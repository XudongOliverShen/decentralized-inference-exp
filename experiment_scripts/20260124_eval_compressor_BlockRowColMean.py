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

def tensor_nbytes(t: torch.Tensor) -> int:
    return t.element_size() * t.numel()


class BlockRowColMean_Compressor(Compressor):
    """
    Blockwise baseline on the last-2 dims (..., N_tokens, M_feats).

    Optional permutations:
      - permute tokens by their mean (ascending) along feature axis (and any leading batch dims).
      - permute features by their mean (ascending) along token axis (and any leading batch dims).
      - store permutations so we can invert them on decompress.

    For each block of shape (Tb, Fb):
      X_hat = row_mean + col_mean - grand_mean   (computed within that block)

    Stores:
      - row_mean per block row: (..., nT, nF, Tb)
      - col_mean per block col: (..., nT, nF, Fb)
      - grand_mean per block:   (..., nT, nF)
      - optional token/feature permutation (N,) / (M,)
    """
    def __init__(
        self,
        token_block: int = 128,
        feat_block: int = 128,
        *,
        permute_tokens_by_mean: bool = False,
        permute_features_by_mean: bool = False,
        store_dtype: torch.dtype = torch.float16,
        compute_dtype: torch.dtype = torch.float32,
    ):
        # allow -1 sentinel
        assert token_block == -1 or token_block > 0
        assert feat_block == -1 or feat_block > 0

        self.token_block = int(token_block)
        self.feat_block = int(feat_block)
        self.permute_tokens_by_mean = bool(permute_tokens_by_mean)
        self.permute_features_by_mean = bool(permute_features_by_mean)
        self.store_dtype = store_dtype
        self.compute_dtype = compute_dtype

        perm_tag = (
            ("ptok" if self.permute_tokens_by_mean else "notok") + "_" +
            ("pfeat" if self.permute_features_by_mean else "nofeat")
        )
        self.name = (
            f"block_rowcol_mean_T{self.token_block}_F{self.feat_block}_"
            f"{perm_tag}_{str(store_dtype).replace('torch.', '')}"
        )

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> Payload:
        assert x.dim() >= 2, "Expected tensor with last-2 dims = (tokens, features)"
        shape = tuple(x.shape)
        N, M = shape[-2], shape[-1]

        # Resolve effective block sizes (-1 => full dimension)
        Tb = N if self.token_block == -1 else self.token_block
        Fb = M if self.feat_block == -1 else self.feat_block

        tok_perm = None
        feat_perm = None

        # ----- optional: permute tokens by (ascending) mean -----
        # token mean: mean over all dims except token dim (-2)
        if self.permute_tokens_by_mean:
            reduce_dims = tuple(d for d in range(x.dim()) if d != (x.dim() - 2))
            tok_mean = x.to(self.compute_dtype).mean(dim=reduce_dims)  # (N,)
            tok_perm = torch.argsort(tok_mean, dim=0).to(torch.int32)  # ascending
            x = x.index_select(-2, tok_perm)

        # ----- optional: permute features by (ascending) mean -----
        # feature mean: mean over all dims except feature dim (-1)
        if self.permute_features_by_mean:
            reduce_dims = tuple(range(0, x.dim() - 1))
            feat_mean = x.to(self.compute_dtype).mean(dim=reduce_dims)  # (M,)
            feat_perm = torch.argsort(feat_mean, dim=0).to(torch.int32) # ascending
            x = x.index_select(-1, feat_perm)

        nT = self._ceil_div(N, Tb)
        nF = self._ceil_div(M, Fb)

        # pad to full blocks on last-2 dims
        padN = nT * Tb - N
        padM = nF * Fb - M
        if padN or padM:
            # pad order: (last_dim_left, last_dim_right, second_last_left, second_last_right)
            x_pad = torch.nn.functional.pad(x, (0, padM, 0, padN))
        else:
            x_pad = x

        # reshape into blocks: (..., nT, Tb, nF, Fb)
        xb = x_pad.to(self.compute_dtype).reshape(shape[:-2] + (nT, Tb, nF, Fb))

        # blockwise means
        row_mean = xb.mean(dim=-1)           # (..., nT, Tb, nF)
        col_mean = xb.mean(dim=-3)           # (..., nT, nF, Fb)

        # reorder row_mean -> (..., nT, nF, Tb)
        row_mean = row_mean.permute(*range(0, row_mean.dim() - 2), -1, -2)

        row_mean = row_mean.to(self.store_dtype)
        col_mean = col_mean.to(self.store_dtype)

        aux = {
            "shape": shape,
            "token_block": Tb,
            "feat_block": Fb,
            "nT": nT,
            "nF": nF,
            "row_mean": row_mean,       # (..., nT, nF, Tb)
            "col_mean": col_mean,       # (..., nT, nF, Fb)
            "permute_tokens_by_mean": self.permute_tokens_by_mean,
            "permute_features_by_mean": self.permute_features_by_mean,
        }
        if tok_perm is not None:
            aux["tok_perm"] = tok_perm.to(torch.int32).cpu()
        if feat_perm is not None:
            aux["feat_perm"] = feat_perm.to(torch.int32).cpu()

        total = tensor_nbytes(row_mean) + tensor_nbytes(col_mean)
        if tok_perm is not None:
            total += tensor_nbytes(aux["tok_perm"])
        if feat_perm is not None:
            total += tensor_nbytes(aux["feat_perm"])

        return Payload(torch.empty(0, device="cpu"), aux, total)

    @torch.no_grad()
    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        shape = p.meta["shape"]
        N, M = shape[-2], shape[-1]

        Tb = int(p.meta["token_block"])
        Fb = int(p.meta["feat_block"])
        nT = int(p.meta["nT"])
        nF = int(p.meta["nF"])

        row_mean = p.meta["row_mean"].to(device=device).to(dtype)       # (..., nT, nF, Tb)
        col_mean = p.meta["col_mean"].to(device=device).to(dtype)       # (..., nT, nF, Fb)

        # 用 row_mean 推导 grand_mean
        # row_mean: (..., nT, nF, Tb)
        grand_mean = row_mean.to(torch.float32).mean(dim=-1).to(dtype)  # (..., nT, nF)

        # reconstruct in block grid: (..., nT, nF, Tb, Fb)
        x_blk = (
            row_mean.unsqueeze(-1) +
            col_mean.unsqueeze(-2) -
            grand_mean.unsqueeze(-1).unsqueeze(-1)
        )

        # permute -> (..., nT, Tb, nF, Fb) then reshape -> (..., nT*Tb, nF*Fb)
        x_blk = x_blk.permute(*range(0, x_blk.dim() - 4), -4, -2, -3, -1)
        x_pad = x_blk.reshape(shape[:-2] + (nT * Tb, nF * Fb))

        # crop back to original size (still possibly permuted)
        x_hat = x_pad[..., :N, :M]

        # ----- invert permutations (reverse order of application) -----
        # We applied token perm first, then feature perm. So invert feature first, then token.
        if p.meta.get("permute_features_by_mean", False):
            feat_perm = p.meta["feat_perm"].to(device=device, dtype=torch.int64)  # (M,)
            feat_inv = torch.empty_like(feat_perm)
            feat_inv[feat_perm] = torch.arange(feat_perm.numel(), device=device, dtype=torch.int64)
            x_hat = x_hat.index_select(-1, feat_inv)

        if p.meta.get("permute_tokens_by_mean", False):
            tok_perm = p.meta["tok_perm"].to(device=device, dtype=torch.int64)  # (N,)
            tok_inv = torch.empty_like(tok_perm)
            tok_inv[tok_perm] = torch.arange(tok_perm.numel(), device=device, dtype=torch.int64)
            x_hat = x_hat.index_select(-2, tok_inv)

        return x_hat



for k in [10000]:
    # for token_block, feat_block in [
    #     (2,2), (2,8), (2,16), 
    #     (8,2), (16, 2), 
    #     (4,4), (6, 6), 
    #     ]:
    for token_block, feat_block in [
        (8,8), (12, 12), (16, 16)
        ]:
        run_ppl_eval(
            wandb_run_name = f"BlockRowColMean_Tb{token_block}Fb{feat_block}",
            dtype = "bf16",
            load_in_8bit = True,
            compressor=BlockRowColMean_Compressor(token_block=token_block, feat_block=feat_block),
            first_k_tokens=k,
            result_dir = str(REPO_ROOT / "results" / f"compressor_BlockRowColMean_Tb{token_block}Fb{feat_block}"),
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()