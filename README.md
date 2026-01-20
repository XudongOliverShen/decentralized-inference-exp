This repository provides a **decentralized / split inference activation compression benchmark**.  
It uses PyTorch forward hooks to intercept hidden states at partition boundaries, **compress** them, **record simulated traffic bytes**, **decompress**, and then continue the forward pass.  
Perplexity (PPL) is evaluated on **WikiText-2**, together with traffic statistics (bytes per token).

You can:
- **Run the eval out of the box** (no code changes required).
- **Implement your own compressor** and evaluate PPL vs. Bytes/Token using the same pipeline.

---

## Environment

- **Python**: 3.9+ (recommended)
- **GPU**: CUDA strongly recommended (CPU works but is very slow)

Install minimal dependencies:

```bash
python -m pip install -U pip
python -m pip install torch transformers datasets tqdm huggingface_hub
python -m pip install wandb
```

If you want 8‑bit / 4‑bit loading via `--load_in_8bit` / `--load_in_4bit`:

```bash
python -m pip install bitsandbytes accelerate
```

---

## Quickstart: run eval (use `eval_compressor_none.py`)

Use `eval_compressor_none.py` as the canonical example and entrypoint. It runs multiple
`first_k_tokens` settings by calling `run_ppl_eval(...)` directly.

### 1) Prepare a local model directory, then provide its path to the eval

`eval_compressor_none.py` calls `run_ppl_eval(model_dir=...)`. You must first **prepare a local
model directory** (in a path of your choice), then **provide that path** to the eval via `model_dir`.

**Important:** `model_dir` should be a **local directory on disk** that contains the model weights
and tokenizer files (i.e., a folder you can pass to `AutoTokenizer.from_pretrained(model_dir)` and
`AutoModelForCausalLM.from_pretrained(model_dir)`).

Example local path:
- `/root/cache/transformers/Qwen/Qwen3-32B`

If you do not have a local folder yet, you can create one with `huggingface_hub`:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-32B', local_dir='/root/cache/transformers/Qwen/Qwen3-32B', local_dir_use_symlinks=False)"
```

### 2) Run the baseline (NoneCompressor)

```bash
python eval_compressor_none.py
```

It will write JSON records under `./results/compressor_none`.

---

## Implementing your own compressor (edit `eval_compressor_none.py`)

### Compressor interface (what you must implement)

The interface is defined in `compressor.py`:
- `Payload(data, meta, nbytes)`:
  - `data`: anything your `decompress` understands (often a tensor or a tuple).
  - `meta`: optional metadata for your scheme.
  - `nbytes`: **simulated traffic bytes** for this transmission (drives Bytes/Token metrics).
- `Compressor.compress(x: torch.Tensor) -> Payload`
- `Compressor.decompress(p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor`

Important requirements:
- **`decompress()` must return a tensor** and should call `.to(device=device, dtype=dtype)` to restore the device/dtype expected by the model.
- **`Payload.nbytes` is what the traffic meter uses**:
  - Compute this according to your compression format (e.g., #bits, packing layout).
  - Approximations are fine as long as they are consistent.

### Minimal template (replace the compressor inside `eval_compressor_none.py`)

To implement your own compressor, copy `eval_compressor_none.py` and edit just two parts:
- **Define your compressor class** (implements `compress()` / `decompress()`).
- **Pass an instance** via `compressor=YourCompressor()`.

Minimal template:

```python
import torch
from compressor import Compressor, Payload
from eval_ppl import run_ppl_eval


class MyCompressor(Compressor):
    name = "my_compressor_v1"

    def compress(self, x: torch.Tensor) -> Payload:
        # Example: cast activations to fp16 as a "compression"
        y = x.to(torch.float16)
        nbytes = y.numel() * y.element_size()  # traffic bytes in this toy scheme
        return Payload(data=y, meta={"orig_dtype": str(x.dtype)}, nbytes=nbytes)

    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Restore tensor to the requested device and dtype
        return p.data.to(device=device, dtype=dtype)


if __name__ == "__main__":
    run_ppl_eval(
        model_name="Qwen/Qwen3-32B",
        model_dir="/root/cache/transformers/Qwen/Qwen3-32B",
        dtype="fp16",
        load_in_8bit=True,
        compressor=MyCompressor(),     # pass an instance
        first_k_tokens=10000,
        batch_windows=2,
        result_dir="./results",
        wandb=True,
        wandb_run_name="MyCompressor_sanity",
    )
```

Run:

```bash
python eval_compressor_none.py
```

## Outputs

- **Console**: live `cur_ppl`, `avg_ppl`, `B/tok`, `bytes`, and `tx` via `tqdm`.
- **JSON**: files like `./results/run_*.json`, including:
  - Average PPL
  - Total NLL and number of loss tokens
  - Traffic totals
  - Per‑link traffic stats
- **Weights & Biases (optional)**:
  - If `--wandb` is enabled (default `True`), logs batch‑level metrics and final metrics under `--wandb_project`.

