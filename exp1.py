#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import math
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm.auto import tqdm

# NEW: local download helper
from huggingface_hub import snapshot_download


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


class Int8SymPerTensor(Compressor):
    """
    Naive baseline: symmetric per-tensor int8.
    TODO: Not verified!
    """
    name = "int8_sym_tensor"
    def compress(self, x: torch.Tensor) -> Payload:
        maxv = x.abs().amax()
        scale = (maxv / 127.0).clamp(min=1e-8)
        q = torch.clamp((x / scale).round(), -127, 127).to(torch.int8)
        nbytes = q.numel() * q.element_size() + 4  # + fp32 scale
        return Payload(q, {"scale": float(scale.detach().cpu().item())}, nbytes)

    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q: torch.Tensor = p.data
        x = q.to(device=device, dtype=torch.float32) * p.meta["scale"]
        return x.to(dtype=dtype)


def make_compressor(name: str) -> Compressor:
    name = name.lower()
    if name in ("none", "identity"):
        return NoneCompressor()
    if name in ("int8", "int8_sym", "int8_sym_tensor"):
        return Int8SymPerTensor()
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
    Split transformer layers across 4 nodes almost equally:
      - node0 hosts embedding + the tail layers (the remainder at the end)
      - node1, node2, node3 host earlier layers
      - output stays on node0

    Example behavior (num_layers=80):
      node1: 0-19, node2: 20-39, node3: 40-59, node0: 60-79
    """
    embed_node, output_node = "node0", "node0"

    n_nodes = 4
    tail_node = "node0"
    mid_nodes = ["node1", "node2", "node3"]

    # Make node0 take the tail "almost-equal" chunk.
    # We assign the first 3 chunks to node1-3, and the remainder (including any extra) to node0.
    base = num_layers // n_nodes          # floor chunk size
    cut1 = base                           # [0, cut1) -> node1
    cut2 = 2 * base                       # [cut1, cut2) -> node2
    cut3 = 3 * base                       # [cut2, cut3) -> node3
    # [cut3, num_layers) -> node0 (tail, includes remainder)

    layer_to_node: List[str] = []
    for i in range(num_layers):
        if i < cut1:
            layer_to_node.append(mid_nodes[0])  # node1



def find_boundaries(embed_node: str, layer_to_node: List[str], output_node: str) -> List[Tuple[str, str, str]]:
    b: List[Tuple[str, str, str]] = []
    if embed_node != layer_to_node[0]:
        b.append(("embed", embed_node, layer_to_node[0]))
    for i in range(len(layer_to_node) - 1):
        if layer_to_node[i] != layer_to_node[i + 1]:
            b.append((f"layer:{i}", layer_to_node[i], layer_to_node[i + 1]))
    if layer_to_node[-1] != output_node:
        b.append((f"layer:{len(layer_to_node)-1}", layer_to_node[-1], output_node))
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

        if hs.dim() >= 2:
            ntokens = int(hs.shape[0] * hs.shape[1])
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
def eval_wikitext2_ppl(
    model,
    tokenizer,
    meter: Optional[TrafficMeter] = None,
    wandb_run: Optional[Any] = None,
    max_length: int = 2048,
    stride: int = 512,
    log_every: int = 1,  # log every N windows
) -> float:
    model.eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(_input_device(model))

    seqlen = ids.size(1)
    nlls = []
    prev_end = 0

    # progress bar over windows
    steps = list(range(0, seqlen, stride))
    pbar = tqdm(steps, desc="wikitext2 ppl", unit="win")

    for step_i, begin in enumerate(pbar, start=1):
        end = min(begin + max_length, seqlen)
        trg_len = end - prev_end

        x = ids[:, begin:end]
        y = x.clone()
        if trg_len < y.size(1):
            y[:, :-trg_len] = -100

        out = model(input_ids=x, labels=y, use_cache=False)
        nlls.append(out.loss * trg_len)

        prev_end = end

        # current ppl estimate
        cur_ppl = float(torch.exp(torch.stack(nlls).sum() / prev_end).item())

        # traffic totals (optional)
        traffic = meter.totals() if meter is not None else None

        # update tqdm postfix
        postfix = {"ppl": f"{cur_ppl:.3f}"}
        if traffic is not None and not math.isnan(traffic["bytes_per_token"]):
            postfix["B/tok"] = f"{traffic['bytes_per_token']:.1f}"
            postfix["bytes"] = f"{int(traffic['total_bytes']):,}"
            postfix["tx"] = f"{int(traffic['total_tx']):,}"
        pbar.set_postfix(postfix)

        # wandb log (optional)
        if wandb_run is not None and (step_i % log_every == 0):
            log_dict = {
                "eval/window": step_i,
                "eval/scored_tokens": prev_end,
                "eval/ppl_running": cur_ppl,
            }
            if traffic is not None:
                log_dict.update({
                    "traffic/total_bytes": traffic["total_bytes"],
                    "traffic/total_tokens": traffic["total_tokens"],
                    "traffic/total_tx": traffic["total_tx"],
                    "traffic/bytes_per_token": traffic["bytes_per_token"],
                })
            wandb_run.log(log_dict)

        if end == seqlen:
            break

    # final ppl
    ppl = float(torch.exp(torch.stack(nlls).sum() / prev_end).item())
    return ppl


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


def prepare_local_model(model_name: str, model_path: Optional[str]) -> str:
    """
    Return the identifier/path to pass to from_pretrained():

    - If model_path is provided:
        * If model_path exists AND looks like a HF model directory -> use it
        * Otherwise -> create it (if needed) and download model_name into it, then use it
    - If model_path is not provided:
        * Load directly from hub using model_name
    """
    if not model_path:
        raise ValueError("Must provide model_path!")

    # If path exists and is a non-empty HF model dir, just use it
    if os.path.isdir(model_path):
        # Heuristics: either config.json exists, or there are model weight shards
        has_config = os.path.isfile(os.path.join(model_path, "config.json"))
        has_weights = any(
            os.path.isfile(os.path.join(model_path, fn))
            for fn in ("model.safetensors", "pytorch_model.bin")
        )
        has_shards = any(
            (fn.startswith("model-") and fn.endswith(".safetensors"))
            or (fn.startswith("pytorch_model-") and fn.endswith(".bin"))
            for fn in os.listdir(model_path)
        )
        if has_config and (has_weights or has_shards):
            return True

    # Otherwise download into model_path (create it if needed)
    os.makedirs(model_path, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=model_path,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return True


# =========================
# 8) Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, default="exp_data", help="Folder to write experiment outputs")

    # default: Qwen3-32B
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")

    # local path + download option
    p.add_argument("--model_path", type=str, default="/datadrive/transformer",
                   help="Local directory to store/load the model. If set, tokenizer/model will load from here.")

    p.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--load_in_4bit", action="store_true", default=False)

    p.add_argument("--compressor", type=str, default="none", choices=["none", "int8"])
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--stride", type=int, default=512)

    # wandb
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="decentralized-infer-compression")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_log_every", type=int, default=100, help="Log every N windows")

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
                "model_path": args.model_path,
                "dtype": args.dtype,
                "load_in_8bit": load8,
                "load_in_4bit": load4,
                "compressor": args.compressor,
                "max_length": args.max_length,
                "stride": args.stride,
            },
        )

    # Decide hub id vs local path
    prepare_local_model(args.model_name, args.model_path)
    print(f"[Model source] {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = load_model(args.model_path, args.dtype, load_in_8bit=load8, load_in_4bit=load4)

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
    ppl = eval_wikitext2_ppl(
        model,
        tokenizer,
        meter=meter,
        wandb_run=wandb_run,
        max_length=args.max_length,
        stride=args.stride,
        log_every=args.wandb_log_every,
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
        "run_id": run_id,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "dtype": args.dtype,
            "load_in_8bit": load8,
            "load_in_4bit": load4,
            "compressor": args.compressor,
            "max_length": args.max_length,
            "stride": args.stride,
        },
        "results": {
            "ppl": float(ppl),
            "seconds": float(t1 - t0),
        },
        "traffic_totals": totals,         # already float values
        "traffic_per_link": meter.stats,  # per-boundary dict
    }

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
