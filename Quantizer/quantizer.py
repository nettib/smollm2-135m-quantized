
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_SCALE = 1e-6          # a scale is never smaller than this (protects against dividing by 0)
SUPPORTED_BITS = (4, 8)



# Step 1: cut the weights into groups

def check_group_size(in_features, group_size):
    if not group_size or in_features % group_size != 0:
        raise ValueError(
            f"group_size={group_size} does not divide in_features={in_features}. "
            f"Pick a group size that divides every layer's input size (for example 32 or 64)."
        )


def num_groups(out_features, in_features, granularity, group_size):
    """How many groups (= how many scales) one weight matrix gets."""
    if granularity == "tensor":
        return 1
    if granularity == "channel":
        return out_features
    if granularity == "group":
        check_group_size(in_features, group_size)
        return out_features * in_features // group_size
    raise ValueError(f"unknown granularity: {granularity}")


def numbers_per_group(out_features, in_features, granularity, group_size):
    return (out_features * in_features) // num_groups(out_features, in_features,
                                                      granularity, group_size)


def split_into_groups(weights, granularity, group_size):
    """[out, in] matrix -> table with one ROW per group: [number of groups, numbers per group]."""
    out_features, in_features = weights.shape
    return weights.reshape(num_groups(out_features, in_features, granularity, group_size), -1)



# Step 2: find the scale and the zero point of every group

def slot_range(bits, mode):
    """Lowest and highest slot we may use.
    asym: 8-bit -> -128..127, 4-bit -> -8..7
    sym : 8-bit -> -127..127, 4-bit -> -7..7   (drops the lowest slot to stay centered on 0)"""
    highest_slot = 2 ** (bits - 1) - 1
    lowest_slot = -highest_slot if mode == "sym" else -highest_slot - 1
    return lowest_slot, highest_slot


def real_range(groups, mode):
    """The smallest and largest real value that each group must be able to store."""
    if mode == "sym":
        biggest = groups.abs().amax(dim=1, keepdim=True)
        return -biggest, biggest
    real_min = groups.amin(dim=1, keepdim=True).clamp(max=0)    # make sure 0 is inside the range
    real_max = groups.amax(dim=1, keepdim=True).clamp(min=0)
    return real_min, real_max


def find_scale_and_zero_point(real_min, real_max, lowest_slot, highest_slot, mode):
    """The two formulas from the top of this file (the same for both modes)."""
    scale = ((real_max - real_min) / (highest_slot - lowest_slot)).clamp(min=MIN_SCALE)
    scale = scale.to(torch.float16).float()     # we save scales as 16-bit numbers, so use that exact value now

    if mode == "sym":
        zero_point = torch.zeros_like(scale)    # the formula below gives exactly 0 here, so skip it
    else:
        zero_point = torch.round(lowest_slot - real_min / scale)
    return scale, zero_point



# Step 3: store the weights, and read them back

def pack4(stored):
    """Put two 4-bit numbers into one byte.  [G, L] values -8..7  ->  [G, L/2] bytes."""
    shifted = (stored + 8).to(torch.uint8)                      # -8..7 becomes 0..15
    return shifted[:, 0::2] | (shifted[:, 1::2] << 4)           # 1st number: low half, 2nd: high half


def unpack4(packed):
    """Opposite of pack4."""
    first = (packed & 0x0F).to(torch.int8) - 8
    second = (packed >> 4).to(torch.int8) - 8
    return torch.stack([first, second], dim=-1).reshape(packed.shape[0], -1)


def quantize_weight(weights, bits=8, mode="asym", granularity="channel", group_size=64):
    """
    Shrink one [out, in] weight matrix. Returns (data, scale, zero_point):
        data        the stored numbers: int8 for 8-bit, or packed bytes for 4-bit
        scale       one number per group (16-bit)
        zero_point  one number per group (asym only; None for sym)
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
    if mode not in ("sym", "asym"):
        raise ValueError("mode must be 'sym' or 'asym'")

    groups = split_into_groups(weights.float(), granularity, group_size)
    lowest_slot, highest_slot = slot_range(bits, mode)
    real_min, real_max = real_range(groups, mode)
    scale, zero_point = find_scale_and_zero_point(real_min, real_max,
                                                  lowest_slot, highest_slot, mode)

    # store = round(weight / scale) + zero_point, kept inside the slots
    stored = torch.round(groups / scale + zero_point).clamp(lowest_slot, highest_slot).to(torch.int8)

    data = pack4(stored) if bits == 4 else stored
    zero_point = None if mode == "sym" else zero_point.to(torch.int8).reshape(-1)
    return data, scale.to(torch.float16).reshape(-1), zero_point


def dequantize_weight(data, scale, zero_point, bits, shape, dtype=torch.float32):
    """Read the weights back: scale * (stored - zero_point)."""
    stored = unpack4(data) if bits == 4 else data
    weights = stored.float()
    if zero_point is not None:
        weights = weights - zero_point.float().unsqueeze(1)
    weights = weights * scale.float().unsqueeze(1)
    return weights.reshape(shape).to(dtype)



# Step 4: a layer that KEEPS the small numbers but CALCULATES in normal precision

class QuantLinear(nn.Module):
    """Replacement for nn.Linear. It stores the shrunken weights, and every time the
    model runs it rebuilds the real weights for this one layer and multiplies as usual."""

    def __init__(self, in_features, out_features, bias, bits, mode, granularity,
                 group_size, dtype=torch.bfloat16):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.mode = bits, mode
        self.granularity, self.group_size = granularity, group_size

        groups = num_groups(out_features, in_features, granularity, group_size)
        length = numbers_per_group(out_features, in_features, granularity, group_size)
        if bits == 4 and length % 2 != 0:
            raise ValueError("4-bit packing needs an even number of weights per group")

        # "buffers" are saved with the model but are not trained
        if bits == 4:
            self.register_buffer("qweight", torch.zeros(groups, length // 2, dtype=torch.uint8))
        else:
            self.register_buffer("qweight", torch.zeros(groups, length, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(groups, dtype=torch.float16))
        if mode == "asym":
            self.register_buffer("zero_point", torch.zeros(groups, dtype=torch.int8))
        else:
            self.register_buffer("zero_point", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear, bits, mode, granularity, group_size):
        """Build a QuantLinear from an ordinary layer by shrinking its weights."""
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None,
                    bits, mode, granularity, group_size, dtype=linear.weight.dtype)
        data, scale, zero_point = quantize_weight(linear.weight.data, bits, mode,
                                                  granularity, group_size)
        layer.qweight, layer.scale = data, scale
        if zero_point is not None:
            layer.zero_point = zero_point
        if linear.bias is not None:
            layer.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        return layer.to(linear.weight.device)

    def dequantized_weight(self, dtype=torch.float32):
        return dequantize_weight(self.qweight, self.scale, self.zero_point, self.bits,
                                 (self.out_features, self.in_features), dtype)

    def forward(self, x):
        weights = self.dequantized_weight(x.dtype)      # rebuild this layer's weights
        return F.linear(x, weights, self.bias)          # then the normal multiplication



# Step 5: shrink a whole model

def quantize_model(model, bits=8, mode="asym", granularity="channel", group_size=64,
                   skip=("lm_head",)):
    """
    Walk through the model and replace every nn.Linear layer with a QuantLinear
    (except the layers named in `skip`: the final output layer stays in full precision).
    The word table (embedding) is not a Linear layer, so it is not shrunk.

    Returns how much the weights changed: weight_mse (average squared difference)
    and weight_mae (average absolute difference).
    """
    total = {"squared": 0.0, "absolute": 0.0, "count": 0, "layers": 0, "skipped": []}

    def visit(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name not in skip:
                original = child.weight.data.float()
                new_layer = QuantLinear.from_linear(child, bits, mode, granularity, group_size)
                difference = original - new_layer.dequantized_weight(torch.float32)
                total["squared"] += difference.pow(2).sum().item()
                total["absolute"] += difference.abs().sum().item()
                total["count"] += difference.numel()
                total["layers"] += 1
                setattr(module, name, new_layer)
            elif isinstance(child, nn.Linear):
                total["skipped"].append(name)
            else:
                visit(child)            # go deeper into nested blocks

    visit(model)
    count = max(total["count"], 1)
    return {"weight_mse": total["squared"] / count, "weight_mae": total["absolute"] / count,
            "quantized_layers": total["layers"], "skipped": total["skipped"]}


def _swap_in_empty_layers(model, cfg, skip=("lm_head",)):
    """Same walk, but the new layers start empty (the weights are loaded from a file afterwards)."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name not in skip:
            empty = QuantLinear(child.in_features, child.out_features, child.bias is not None,
                                cfg["bits"], cfg["mode"], cfg["granularity"], cfg["group_size"],
                                dtype=child.weight.dtype)
            setattr(model, name, empty)
        elif not isinstance(child, nn.Linear):
            _swap_in_empty_layers(child, cfg, skip)



# Step 6: measure sizes

def weights_mb(model):
    """Memory used by all the model's stored numbers, in MB (a tensor shared by two layers counts once)."""
    seen, total_bytes = set(), 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        address = tensor.data_ptr()
        if address in seen:
            continue
        seen.add(address)
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / 1e6


def save_model_file(model, path):
    """Save the model to one .safetensors file and return the file size in MB."""
    from safetensors.torch import save_model
    save_model(model, path)
    return os.path.getsize(path) / 1e6



# Step 7: save a shrunken model and load it again (this is what goes on Hugging Face)

def save_quantized(model, tokenizer, out_dir, base_model_id, cfg, compute_dtype="bfloat16"):
    os.makedirs(out_dir, exist_ok=True)
    save_model_file(model, os.path.join(out_dir, "model.safetensors"))
    model.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    settings = dict(cfg, base_model_id=base_model_id, compute_dtype=compute_dtype, skip=["lm_head"])
    with open(os.path.join(out_dir, "quant_config.json"), "w") as f:
        json.dump(settings, f, indent=2)


def load_quantized(path_or_repo, device="cpu"):
    """Load a model saved by save_quantized (from a folder, or from a Hugging Face repo name)."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    folder = path_or_repo if os.path.isdir(path_or_repo) else snapshot_download(path_or_repo)
    with open(os.path.join(folder, "quant_config.json")) as f:
        settings = json.load(f)
    dtype = getattr(torch, settings["compute_dtype"])

    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(folder)).to(dtype)
    _swap_in_empty_layers(model, settings, tuple(settings.get("skip", ["lm_head"])))
    load_model(model, os.path.join(folder, "model.safetensors"))
    return model.to(device).eval(), AutoTokenizer.from_pretrained(folder)
