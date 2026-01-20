This README provides an overview of the **Decentralized Inference Compression Benchmark**. This tool is designed to evaluate the impact of different activation compression methods on model perplexity (PPL) while simulating the network traffic overhead of a partitioned LLM.

---

# Decentralized Inference Compression Benchmark

This repository contains a framework to simulate "split inference" across multiple nodes. It uses PyTorch forward hooks to intercept hidden states at specific layer boundaries, compresses them, records the simulated "wire" size, decompresses them, and continues the forward pass.

## 🚀 1. Environment Setup

To run this project, you need Python 3.8+ and a CUDA-capable environment. For 8-bit or 4-bit weight quantization, `bitsandbytes` and `accelerate` are required. It uses `wandb` for logging.

```bash
# Install core dependencies
pip install torch
pip install transformers datasets tqdm wandb

# Install requirements for 4-bit/8-bit loading
pip install bitsandbytes accelerate

```

---

## 📂 2. Model Management & Performance

The script includes a helper to manage local model storage. On "Glows" machines, reading from datadrive mounts like `/datadrive` is slow. So I do below worflow:

1. Store large model weights/cache in a persistent but slower directory (e.g., `/datadrive/cache`).
2. At the start of my session, move the cache folder to high-speed local storage (e.g., `/root/cache/`).
3. Use the `--model_dir` argument to point to this high-speed location.

---

## 🏃 3. Running the Benchmark

The main script is `eval_ppl_batch.py`. It evaluates perplexity on **WikiText-2**.

### Basic Usage

```bash
python3 eval_ppl_batch.py \
    --model_name Qwen/Qwen3-32B \
    --model_dir /root/cache/transformers \
    --dtype fp16 \
    --load_in_8bit \
    --compressor none \
    --batch_size 2 \
    --first_k_tokens 10000

```

### Key Arguments

* `--first_k_tokens`: Sets the evaluation length. Full WikiText-2 (~299k tokens) takes about 9 minutes on an L40S.
* **Quick Test:** `10000`, `20000`, `50000`, or `100000` tokens.
* **Full Eval:** `0` (uses all tokens).


* `--compressor`: Default is none. Implement new compressor to run eval.
* `--load_in_4bit` or `--load_in_8bit`: Enables weight quantization to fit large models on smaller VRAM. Let's use 8bit for now.
* `--batch_size`: let's use 2 for now. different batch size leads to small change in perpleixty.

---

## 🛠️ 4. Implementing New Compression Methods

You can extend the framework by subclassing the `Compressor` and `Payload` classes. This allows you to test custom algorithms (e.g., FP8, pruning, or non-linear quantization).

### The Interface

To add a new method, implement the `compress` and `decompress` methods:

```python
class YourCustomCompressor(Compressor):
    name = "custom_name"

    def compress(self, x: torch.Tensor) -> Payload:
        # 1. Apply your compression logic
        # 2. Calculate the size in bytes for traffic simulation
        nbytes = x.numel() * 1 # Example: 1 byte per element
        return Payload(data=compressed_data, meta={"scale": 1.0}, nbytes=nbytes)

    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # 1. Recover the tensor
        # 2. Ensure it returns to the original device and dtype
        return recovered_tensor.to(device=device, dtype=dtype)

```

After implementing, register it in the `make_compressor` factory function.

---

## 📊 5. Simulation Logic: Partitioning

The script uses a "Default Plan" to simulate a 4-node cluster:

1. **Node 0:** Embedding + Early Layers + Late Layers + Output Head.
2. **Node 1-3:** Middle Layers.

Communication (and thus compression + decompression) is triggered whenever the "node" assignment changes between two layers.

---

## 📈 6. Outputs

* **Console:** Real-time PPL and `Bytes/Token` metrics via tqdm.
* **JSON Record:** A full summary is saved in `--exp_dir` including per-link traffic statistics and total NLL.
* **WandB:** Provides live curves of window-based PPL vs. total tokens processed.
