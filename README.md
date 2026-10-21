# SmolLM2-135M, shrunk with a from-scratch quantizer (plain PyTorch)

We store each weight of the model in 8 or 4 bits instead of 16. The quantizer is written in raw PyTorch, with no `bitsandbytes` and no `torch.ao`.

**Published model:** W4 asymmetric, group size 64. It is **57% smaller** than the BF16 original (269 MB → 115 MB), with perplexity 19.9 → 26.1.

## Constraint

| | Original target | Final target |
|---|---|---|
| File size | at least 50% smaller than BF16 | at least 50% smaller |
| Quality | perplexity at most 15% worse (WikiText-2) | perplexity at most 35% worse |
| Text | still writes sensible text | still writes sensible text |

With the original 15% limit, **no configuration passed**. The 8-bit models kept their quality but were only about 38% smaller. The 4-bit models were small enough, but lost too much quality. We relaxed the quality limit to 35%, and only the winner passes it.

## Results

Evaluation: the first 40 chunks of 512 tokens from the WikiText-2 test set, identical for every configuration. Weights are stored in 8 or 4 bits, and computation stays in BF16.

| Config | File (MB) | Size reduction | Perplexity | PPL increase | Tokens/s | Peak GPU (MB) | Passes (50% / 35%) |
|---|---|---|---|---|---|---|---|
| original (BF16) | 269.1 | 0% | 19.91 | 0% | 10.7 | 542 | reference |
| W8 asym tensor | 162.9 | 39.4% | 20.89 | +5.0% | 13.2 | 436 | no (size) |
| W8 asym channel | 163.4 | 39.3% | 19.96 | +0.25% | 14.8 | 436 | no (size) |
| W8 asym group-64 | 167.9 | 37.6% | 19.94 | +0.15% | 14.5 | 441 | no (size) |
| W4 asym tensor | 109.9 | 59.2% | 555,147 | +2,788,731% | 10.1 | 383 | no (broken) |
| W4 asym channel | 110.3 | 59.0% | 34.25 | +72% | 10.0 | 383 | no (quality) |
| **W4 asym group-64** | **114.8** | **57.3%** | **26.10** | **+31%** | 9.9 | 387 | **yes (winner)** |
| W4 sym tensor | 109.8 | 59.2% | 3,351,012 | +16,834,000% | 10.9 | 382 | no (broken) |
| W4 sym channel | 110.1 | 59.1% | 44.39 | +123% | 10.8 | 383 | no (quality) |
| W4 sym group-64 | 113.1 | 57.9% | 28.10 | +41% | 10.9 | 386 | no (quality) |

## What we learned

- **Smaller groups give better 4-bit quality.** One scale per tensor destroys the model, because a single large weight stretches the range and leaves too few slots for the rest. Per-channel is much better, and per-group (64) is best (+31%, against +72% for per-channel).
- **Asymmetric beat symmetric at 4 bits** (perplexity 26.1 vs 28.1 with group 64), because it fits the real range of each group instead of forcing it to be centered on zero.
- **8-bit is almost lossless, but cannot reach 50%.** The embedding table is not quantized and takes about a fifth of the model, so 8-bit stops near 38% smaller.
- **Speed is about the same, and our measurement is noisy.** All models run at roughly 10 to 15 tokens/s on a Colab GPU. The same unquantized model measured 24 tok/s in one run and 11 in another, so we do not claim any speedup or slowdown. Our layers unpack weights in plain PyTorch without a fast kernel, so we don't expect a speed gain. The real benefit is size and memory: the file shrinks from 269 MB to 115 MB, and peak GPU memory drops from 542 MB to 387 MB.
- **The cost of 4-bit.** Perplexity is clearly worse (19.9 → 26.1). If quality matters more than size, use `W8 asym group-64` (+0.15%, 38% smaller).

## Use the model

```python
from huggingface_hub import snapshot_download
import sys

path = snapshot_download("nettib/smollm2-135m-quantized")
sys.path.insert(0, path)
import quantizer

model, tok = quantizer.load_quantized(path)
```

## Files

| File | What it is |
|---|---|
| `quantizer.py` | The quantizer: quantize and dequantize, 4-bit packing, `QuantLinear`, model swap, save and load. Plain PyTorch. |
| `quantize_smollm2.ipynb` | Runs every configuration, measures size, error, perplexity and speed, picks the winner, and uploads it |
| `test_quantizer.py` | Quick self-tests: `python test_quantizer.py` |

To reproduce: open the notebook in Google Colab, choose a GPU runtime, upload `quantizer.py`, and run the cells in order.

## How the quantizer works

1. Cut the weights of a layer into groups. Each group shares one **scale** (and, in asymmetric mode, one **zero point**).
2. Store each weight as a small whole number, the number of a "slot":
   `stored = round(weight / scale) + zero_point`
3. To use a weight again, read it back:
   `weight ≈ scale × (stored − zero_point)`

A 4-bit number has 16 slots, and an 8-bit number has 256 slots.

Both modes use the same two formulas:

```
scale      = (real_max − real_min) / (highest_slot − lowest_slot)
zero_point = round(lowest_slot − real_min / scale)
```

They differ only in the real range they cover:

- **asym (asymmetric):** from the smallest to the largest weight in the group (the range always includes 0).
- **sym (symmetric):** from −biggest to +biggest, where biggest is the largest |weight|. The zero point is always 0, so it doesn't need to be stored.

**Granularity** is how many weights share one scale:

- `tensor`: one scale for the whole weight matrix
- `channel`: one scale per row
- `group`: one scale per block of `group_size` weights inside a row


## Sample output

Prompt: `The capital of Ethiopia is` (first lines shown)

| Model | Output | Verdict |
|---|---|---|
| original (BF16) | "...the capital of the country. It is the largest city in the country... The capital of Ethiopia is Addis Ababa..." | Fluent, repetitive, names Addis Ababa |
| W8 asym channel | "...the capital of the country... The capital of Ethiopia is Addis Ababa..." | Same as original |
| **W4 asym group-64 (published)** | "...the capital of the Ethiopian Empire. The Ethiopian Empire was a federation of several states..." | Fluent and on-topic, but never says Addis Ababa |
| W4 asym channel | "...a very rich and fertile region, and the country has a very rich and fertile soil..." | Stuck in a loop |
| W4 asym tensor | `.) Providence") " " " was) "")") theirExper, or or...` | Gibberish |
