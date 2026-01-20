#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import math
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm.auto import tqdm

# NEW: local download helper
from huggingface_hub import snapshot_download


def format_bytes(n: float, binary: bool = False) -> str:
    """
    Format bytes using K/M/G/T suffix.
    binary=False -> 1K = 1000
    binary=True  -> 1Ki = 1024
    """
    if n < 0:
        return "-" + format_bytes(-n, binary)

    base = 1024 if binary else 1000
    suffixes = ["B", "K", "M", "G", "T", "P"] if not binary else ["B", "Ki", "Mi", "Gi", "Ti", "Pi"]

    for suf in suffixes:
        if n < base:
            return f"{n:.1f}{suf}" if suf != "B" else f"{int(n)}{suf}"
        n /= base
    return f"{n:.1f}{suffixes[-1]}"


def format_duration(seconds: float) -> str:
    # seconds can be float
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)

    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)

    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s {ms:03d}ms"


# =========================
# 1) Compressor
# =========================
@dataclass
class Payload:
    data: Any
    meta: Dict[str, Any]
    nbytes: int  # simulated traffic bytes


class Compressor:
    name = "base"
    def compress(self, x: torch.Tensor) -> Payload: raise NotImplementedError
    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor: raise NotImplementedError


class NoneCompressor(Compressor):
    name = "none"
    def compress(self, x: torch.Tensor) -> Payload:
        return Payload(x, {}, x.numel() * x.element_size())
    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return p.data.to(device=device)

def make_compressor(name: str) -> Compressor:
    name = name.lower()
    if name in ("none", "identity"):
        return NoneCompressor()
    raise ValueError(f"Unknown compressor: {name}")


# =========================
# 2) Traffic meter
# =========================
class TrafficMeter:
    def __init__(self):
        # per-link stats
        self.stats: Dict[str, Dict[str, int]] = {}
        # global totals (streaming)
        self.total_bytes = 0
        self.total_tokens = 0
        self.total_tx = 0

    def record(self, key: str, nbytes: int, ntokens: int):
        nb = int(nbytes)
        nt = int(ntokens)

        # update per-key
        st = self.stats.get(key)
        if st is None:
            st = {"bytes": 0, "tokens": 0, "tx": 0}
            self.stats[key] = st
        st["bytes"] += nb
        st["tokens"] += nt
        st["tx"] += 1

        # update global (O(1))
        self.total_bytes += nb
        self.total_tokens += nt
        self.total_tx += 1

    def totals(self) -> Dict[str, float]:
        bpt = (self.total_bytes / self.total_tokens) if self.total_tokens else float("nan")
        return {
            "total_bytes": float(self.total_bytes),
            "total_tokens": float(self.total_tokens),
            "total_tx": float(self.total_tx),
            "bytes_per_token": float(bpt),
        }

    def reset(self):
        self.stats.clear()
        self.total_bytes = 0
        self.total_tokens = 0
        self.total_tx = 0


# =========================
# 3) Pipeline: compress -> record -> decompress
# =========================
class Pipeline:
    def __init__(self, compressor: Compressor, meter: TrafficMeter):
        self.compressor = compressor
        self.meter = meter

    def run(self, x: torch.Tensor, *, key: str, ntokens: int) -> torch.Tensor:
        p = self.compressor.compress(x)                          # 1) compress
        self.meter.record(key, p.nbytes, ntokens)                # 2) record
        return self.compressor.decompress(p, x.device, x.dtype)  # 3) decompress


# =========================
# 4) Partition -> boundaries
# =========================
def make_default_plan(num_layers: int) -> Tuple[str, List[str], str]:
    """
    4 nodes, almost equal layers.
      - node0 hosts embedding + its share of layers split into early+late halves
      - when uneven, extra layers go to node1/2/3 (not node0)
      - output on node0
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}")

    embed_node = output_node = "node0"

    base = num_layers // 4
    rem  = num_layers % 4

    # node0 gets the base share; extra goes to node1/2/3
    c0 = base
    c1 = base + (1 if rem >= 1 else 0)
    c2 = base + (1 if rem >= 2 else 0)
    c3 = base + (1 if rem >= 3 else 0)

    e = c0 // 2          # node0 early
    l = c0 - e           # node0 late

    layer_to_node: List[str] = [""] * num_layers
    for i in range(e):
        layer_to_node[i] = "node0"
    for i in range(num_layers - l, num_layers):
        layer_to_node[i] = "node0"

    mid = [i for i, v in enumerate(layer_to_node) if not v]
    k = 0
    for _ in range(c1):
        layer_to_node[mid[k]] = "node1"; k += 1
    for _ in range(c2):
        layer_to_node[mid[k]] = "node2"; k += 1
    for _ in range(c3):
        layer_to_node[mid[k]] = "node3"; k += 1

    return embed_node, layer_to_node, output_node



def find_boundaries(embed_node: str, layer_to_node: List[str], output_node: str) -> List[Tuple[str, str, str]]:
    """
    Return boundary hops as (where, src, dst).

    Boundaries include:
      - embed -> first layer node (if different)
      - layer i -> layer i+1 (if node changes)
      - last layer -> output node (if different)
      - output node -> embed node (if different), because generation needs output on embed node
    """
    b: List[Tuple[str, str, str]] = []
    if not layer_to_node:
        return b

    if embed_node != layer_to_node[0]:
        b.append(("embed", embed_node, layer_to_node[0]))

    for i in range(len(layer_to_node) - 1):
        if layer_to_node[i] != layer_to_node[i + 1]:
            b.append((f"layer:{i}", layer_to_node[i], layer_to_node[i + 1]))

    last_where = f"layer:{len(layer_to_node) - 1}"
    if layer_to_node[-1] != output_node:
        b.append((last_where, layer_to_node[-1], output_node))

    # extra hop for generation: output must be available back on embed node
    if output_node != embed_node:
        b.append(("output", output_node, embed_node))

    return b


# =========================
# 5) Hooks (short)
# =========================
def make_hook(pipeline: Pipeline, key: str):
    """
    forward hook: compress -> record -> decompress
    supports out as: Tensor or tuple(out[0]=hs)
    """
    def _hook(module, inp, out):
        if torch.is_tensor(out):
            hs, pack = out, (lambda new: new)
        elif isinstance(out, tuple) and out and torch.is_tensor(out[0]):
            hs, pack = out[0], (lambda new: (new,) + out[1:])
        else:
            return out

        if hs.dim() == 3:
            ntokens = int(hs.shape[0] * hs.shape[1])   # B*T
        elif hs.dim() == 2:
            ntokens = int(hs.shape[0])                 # T
        else:
            raise ValueError(f"Unrecognized hidden state of shape {hs.shape}, will lead to wrong token count.")

        return pack(pipeline.run(hs, key=key, ntokens=ntokens))
    return _hook


def install_boundary_hooks(model, boundaries: List[Tuple[str, str, str]], pipeline: Pipeline):
    handles = []
    embed = model.model.embed_tokens
    layers = model.model.layers
    for where, src, dst in boundaries:
        key = f"{where}:{src}->{dst}:{pipeline.compressor.name}"
        if where == "embed":
            handles.append(embed.register_forward_hook(make_hook(pipeline, key)))
        elif where.startswith("layer:"):
            i = int(where.split(":")[1])
            handles.append(layers[i].register_forward_hook(make_hook(pipeline, key)))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# =========================
# 6) WikiText2 PPL (with tqdm + wandb logging)
# =========================
def _input_device(model) -> torch.device:
    # for device_map="auto": inputs should be on embed device
    try:
        return model.model.embed_tokens.weight.device
    except Exception:
        return next(model.parameters()).device



@torch.no_grad()
def eval_wikitext2_ppl_serial(
    model,
    tokenizer,
    meter: Optional[TrafficMeter] = None,
    wandb_run: Optional[Any] = None,
    max_length: int = 2048,
    stride: int = 512,
    log_every: int = 1,
    first_k_tokens: int = 0,
) -> float:
    """
    deprecated
    Use eval_wikitext2_ppl instead for batch inference
    """
    model.eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(_input_device(model))

    if first_k_tokens and first_k_tokens > 0:
        ids = ids[:, : min(first_k_tokens, ids.size(1))]

    seqlen = ids.size(1)
    prev_end = 0

    total_nll = 0.0
    total_loss_tokens = 0

    steps = list(range(0, seqlen, stride))
    pbar = tqdm(steps, desc="wikitext2 ppl", unit="win")

    for step_i, begin in enumerate(pbar, start=1):
        end = min(begin + max_length, seqlen)
        trg_len = end - prev_end  # 这一段代码中真正属于当前步要预测的长度

        x = ids[:, begin:end]
        y = x.clone()
        
        # Mask 掉不属于当前 stride 的部分（即属于 context 的部分）
        if trg_len < y.size(1):
            y[:, :-trg_len] = -100

        out = model(input_ids=x, labels=y, use_cache=False)
        
        # 1. 找到所有非 -100 的 label 数量
        # HF 实际用的是 shift_labels = y[:, 1:]
        num_loss_tokens = (y[:, 1:] != -100).sum().item()

        # 此时 out.loss 是对所有非 -100 位的平均，我们需要还原回 sum
        # 虽然 HuggingFace 的内部计算稍显复杂，但在滑动窗口下：
        # 总 NLL = 平均 Loss * 实际参与计算的有效 Token 数
        window_avg_nll = float(out.loss.item())
        total_nll += window_avg_nll * num_loss_tokens
        total_loss_tokens += num_loss_tokens
        
        # 计算当前平均指标
        avg_ppl = math.exp(total_nll / total_loss_tokens)
        window_ppl = math.exp(window_avg_nll)

        prev_end = end

        # traffic totals (optional)
        traffic = meter.totals() if meter is not None else None

        # update tqdm postfix
        postfix = {
            "cur_ppl": f"{window_ppl:.2f}",
            "avg_ppl": f"{avg_ppl:.2f}",
        }
        if traffic is not None and not math.isnan(traffic["bytes_per_token"]):
            postfix["B/tok"] = f"{traffic['bytes_per_token']:.1f}"
            postfix["bytes"] = format_bytes(traffic["total_bytes"])
            postfix["tx"] = f"{int(traffic['total_tx']):,}"
        pbar.set_postfix(postfix)

        # wandb log (optional)
        if wandb_run is not None and (step_i % log_every == 0):
            log_dict = {
                "eval_window/window": step_i,
                "eval_window/window_tokens": num_loss_tokens,
                "eval_window/window_avg_nll": window_avg_nll,
                "eval_window/window_avg_ppl": window_ppl,
                "eval_total/total_tokens": total_loss_tokens,
                "eval_total/total_nll": total_nll,
                "eval_total/total_avg_ppl": avg_ppl,
            }
            if traffic is not None:
                log_dict.update({
                    "traffic/total_bytes": traffic["total_bytes"],
                    "traffic/total_tokens": traffic["total_tokens"],
                    "traffic/total_tx": traffic["total_tx"],
                    "traffic/bytes_per_token": traffic["bytes_per_token"],
                })
            wandb_run.log(log_dict, step=total_loss_tokens)

        if end == seqlen:
            break

    # final ppl
    ppl = math.exp(total_nll / total_loss_tokens)
    return ppl, total_nll, total_loss_tokens


@torch.no_grad()
def eval_wikitext2_ppl(
    model,
    tokenizer,
    meter: Optional["TrafficMeter"] = None,
    wandb_run: Optional[Any] = None,
    max_length: int = 2048,
    stride: int = 512,
    log_every: int = 1,          # 每 N 个 batch log 一次
    first_k_tokens: int = 0,
    batch_windows: int = 1,      # 每个 batch 包含多少个 window
):
    """
    Batch-able WikiText2 perplexity evaluation.
    - 每个 batch forward 一次
    - tqdm / wandb 每个 batch 更新一次
    - wandb 的 step 使用 batch_idx: 0,1,2,...

    Returns:
      ppl, total_nll, total_loss_tokens
    """
    model.eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(_input_device(model))

    if first_k_tokens and first_k_tokens > 0:
        ids = ids[:, : min(first_k_tokens, ids.size(1))]

    seqlen = ids.size(1)

    # 生成窗口列表 (begin, end, trg_len)，保持与 batch=1 完全一致的 trg_len 逻辑
    begins = list(range(0, seqlen, stride))
    windows = []
    prev_end = 0
    for begin in begins:
        end = min(begin + max_length, seqlen)
        trg_len = end - prev_end
        windows.append((begin, end, trg_len))
        prev_end = end
        if end == seqlen:
            break

    # pad token：没有 pad_token_id 的 tokenizer 用 eos 兜底
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    total_nll = 0.0
    total_loss_tokens = 0

    batch_starts = list(range(0, len(windows), batch_windows))
    pbar = tqdm(batch_starts, desc="wikitext2 ppl", unit="batch")

    for batch_idx, batch_start in enumerate(pbar):
        batch = windows[batch_start: batch_start + batch_windows]
        B = len(batch)
        if B == 0:
            break

        # pad 到本 batch 最大长度
        lens = [end - begin for (begin, end, _) in batch]
        L = max(lens)

        x = torch.full((B, L), pad_id, dtype=ids.dtype, device=ids.device)
        attn = torch.zeros((B, L), dtype=torch.long, device=ids.device)
        y = torch.full((B, L), -100, dtype=ids.dtype, device=ids.device)

        # 组装 input / labels
        for b, (begin, end, trg_len) in enumerate(batch):
            l = end - begin
            xb = ids[:, begin:end].squeeze(0)  # [l]

            x[b, :l] = xb
            attn[b, :l] = 1

            yb = xb.clone()
            # 只让最后 trg_len tokens 参与 loss
            yb[:-trg_len] = -100
            y[b, :l] = yb

        # forward：不传 labels，手动算 per-example NLL（等价 HF shift + ignore_index=-100）
        out = model(input_ids=x, attention_mask=attn, use_cache=False)
        logits = out.logits  # [B, L, V]

        shift_logits = logits[:, :-1, :]   # [B, L-1, V]
        shift_labels = y[:, 1:]            # [B, L-1]
        valid_mask = (shift_labels != -100)

        shift_logits = logits[:, :-1, :]          # [B, L-1, V]
        shift_labels = y[:, 1:]                   # [B, L-1]

        V = shift_logits.size(-1)
        loss_flat = F.cross_entropy(
            shift_logits.reshape(-1, V).float(),  # fp32 compute
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        batch_nll = float(loss_flat.sum().item())
        batch_tok = int((shift_labels != -100).sum().item())

        total_nll += batch_nll
        total_loss_tokens += batch_tok

        # 每个 batch 更新一次 tqdm / wandb
        if batch_tok > 0:
            batch_avg_nll = batch_nll / batch_tok
            batch_ppl = math.exp(batch_avg_nll)
        else:
            batch_avg_nll = float("nan")
            batch_ppl = float("nan")

        avg_ppl = math.exp(total_nll / total_loss_tokens) if total_loss_tokens > 0 else float("nan")

        traffic = meter.totals() if meter is not None else None

        postfix = {
            "batch": f"{batch_idx}",
            "cur_ppl": f"{batch_ppl:.2f}",
            "avg_ppl": f"{avg_ppl:.2f}",
        }
        if traffic is not None and not math.isnan(traffic["bytes_per_token"]):
            postfix["B/tok"] = f"{traffic['bytes_per_token']:.1f}"
            postfix["bytes"] = format_bytes(traffic["total_bytes"])
            postfix["tx"] = f"{int(traffic['total_tx']):,}"
        pbar.set_postfix(postfix)

        # wandb：每 N 个 batch log 一次
        if wandb_run is not None and log_every > 0 and ((batch_idx + 1) % log_every == 0):
            log_dict = {
                "eval_batch/batch_idx": batch_idx,
                "eval_batch/batch_tokens": batch_tok,
                "eval_batch/batch_avg_nll": batch_avg_nll,
                "eval_batch/batch_ppl": batch_ppl,

                "eval_total/total_tokens": total_loss_tokens,
                "eval_total/total_nll": total_nll,
                "eval_total/total_avg_nll": (total_nll / total_loss_tokens) if total_loss_tokens > 0 else float("nan"),
                "eval_total/total_avg_ppl": avg_ppl,
            }
            if traffic is not None:
                log_dict.update({
                    "traffic/total_bytes": traffic["total_bytes"],
                    "traffic/total_tokens": traffic["total_tokens"],
                    "traffic/total_tx": traffic["total_tx"],
                    "traffic/bytes_per_token": traffic["bytes_per_token"],
                })

            # 关键：wandb step 用 batch_idx（0,1,2,3...）
            wandb_run.log(log_dict, step=batch_idx)

    ppl = math.exp(total_nll / total_loss_tokens) if total_loss_tokens > 0 else float("nan")
    return ppl, total_nll, total_loss_tokens























# =========================
# 7) Model loading (dtype arg + optional 8/4bit)
# =========================
def _dtype_from_str(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[s]


def load_model(model_id_or_path: str, dtype_str: str, load_in_8bit: bool, load_in_4bit: bool):
    dtype = _dtype_from_str(dtype_str)
    kwargs = {"device_map": "auto"} if torch.cuda.is_available() else {}

    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig  # needs: bitsandbytes + accelerate
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=bool(load_in_8bit),
            load_in_4bit=bool(load_in_4bit),
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype

    return AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)

def prepare_local_model(model_name: str, model_dir: Optional[str]) -> str:
    """
    Ensure the model is available locally under:
        <model_dir>/<model_name>

    Returns the local path to pass to from_pretrained().

    Behavior:
      - model_dir must be provided
      - target_path = os.path.join(model_dir, model_name)  (keeps org/name layout)
      - If target_path already looks like a HF model directory -> return target_path
      - Otherwise download model_name into target_path and return it
    """
    if not model_dir:
        raise ValueError("Must provide model_dir!")

    target_path = os.path.join(model_dir, model_name)

    def looks_like_hf_model_dir(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        has_config = os.path.isfile(os.path.join(path, "config.json"))
        if not has_config:
            return False

        # Common weight file patterns
        direct_weights = ("model.safetensors", "pytorch_model.bin")
        has_weights = any(os.path.isfile(os.path.join(path, fn)) for fn in direct_weights)

        # Sharded weight patterns
        try:
            files = os.listdir(path)
        except OSError:
            return False

        has_shards = any(
            (fn.startswith("model-") and fn.endswith(".safetensors")) or
            (fn.startswith("pytorch_model-") and fn.endswith(".bin"))
            for fn in files
        )
        return has_weights or has_shards

    if looks_like_hf_model_dir(target_path):
        return target_path

    os.makedirs(target_path, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=target_path,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return target_path


# =========================
# 8) Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, default="/root/work/decentralized-inference-exp/exp_results/sanity_runs", help="Folder to write experiment outputs")

    # default: Qwen3-32B
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")

    # local path + download option
    p.add_argument("--model_dir", type=str, default="/root/cache/transformers",
                   help="Local directory to store/load the model. If set, tokenizer/model will load from here.")

    p.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--batch_size", type=int, default=2)

    p.add_argument("--compressor", type=str, default="none", choices=["none"])
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--first_k_tokens", type=int, default=5000,
               help="Only evaluate perplexity on the first K tokens of WikiText-2 test. 0 = full length. # tokens of WikiText-2 test is 299078")


    # wandb
    p.add_argument("--wandb", action="store_true", default=True, help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="decentralized-infer-compression")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_log_every", type=int, default=10, help="Log every N windows")

    return p.parse_args()


def main():
    args = parse_args()
    load4 = bool(args.load_in_4bit)
    load8 = bool(args.load_in_8bit) and (not load4)

    # wandb init (optional)
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_name": args.model_name,
                "model_dir": args.model_dir,
                "dtype": args.dtype,
                "load_in_8bit": load8,
                "load_in_4bit": load4,
                "compressor": args.compressor,
                "max_length": args.max_length,
                "stride": args.stride,
                "first_k_tokens": args.first_k_tokens,
                "batch_size": args.batch_size
            },
        )

    # Decide hub id vs local path
    model_path = model_path = prepare_local_model(args.model_name, args.model_dir)
    print(f"[Model source] {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = load_model(model_path, args.dtype, load_in_8bit=load8, load_in_4bit=load4)

    num_layers = len(model.model.layers)
    print(f"Loaded layers: {num_layers}")
    print(f"Weight quant: {'4bit' if load4 else '8bit' if load8 else 'none'} | dtype={args.dtype}")

    embed_node, layer_to_node, output_node = make_default_plan(num_layers)

    boundaries = find_boundaries(embed_node, layer_to_node, output_node)
    print(f"Boundaries: {boundaries}")

    compressor = make_compressor(args.compressor)
    meter = TrafficMeter()
    pipeline = Pipeline(compressor, meter)

    handles = install_boundary_hooks(model, boundaries, pipeline)
    print(f"\n=== With hooks: {compressor.name} ===")

    t0 = time.time()
    ppl, total_nll, total_loss_tokens = eval_wikitext2_ppl(
        model,
        tokenizer,
        meter=meter,
        wandb_run=wandb_run,
        max_length=args.max_length,
        stride=args.stride,
        log_every=args.wandb_log_every,
        first_k_tokens=args.first_k_tokens, 
        batch_windows=args.batch_size,
    )
    t1 = time.time()
    remove_hooks(handles)

    totals = meter.totals()
    print(f"\nPPL: {ppl:.4f} | time: {t1 - t0:.1f}s")
    print("\n--- Traffic ---")
    print(json.dumps(meter.totals(), indent=2, ensure_ascii=False))

    # ---- write output file (experiment record) ----
    os.makedirs(args.exp_dir, exist_ok=True)

    # 用时间戳做文件名，避免覆盖
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.exp_dir, f"run_{run_id}.json")

    record = {
        "script_name": "eval_ppl_batch.py",
        "run_id": run_id,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_run_s": t1 - t0,
        "time_run_hms": format_duration(t1 - t0),  # readable
        "args": {
            "model_name": args.model_name,
            "model_dir": args.model_dir,
            "dtype": args.dtype,
            "load_in_8bit": load8,
            "load_in_4bit": load4,
            "compressor": args.compressor,
            "max_length": args.max_length,
            "stride": args.stride,
            "first_k_tokens": args.first_k_tokens,
            "batch_size": args.batch_size,
        },
        "results": {
            "avg_ppl": float(ppl),
            "total_nll": total_nll,
            "total_loss_tokens": total_loss_tokens,
        },
        "traffic_totals": totals,         # already float values
        "traffic_per_link": meter.stats,  # per-boundary dict
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] experiment record -> {out_path}")

    # final wandb log (optional)
    if wandb_run is not None:
        wandb_run.log({
            "eval/ppl_final": ppl,
            "traffic/total_bytes": totals["total_bytes"],
            "traffic/total_tokens": totals["total_tokens"],
            "traffic/total_tx": totals["total_tx"],
            "traffic/bytes_per_token": totals["bytes_per_token"],
            "eval/seconds": (t1 - t0),
        })
        wandb_run.finish()


if __name__ == "__main__":
    main()
