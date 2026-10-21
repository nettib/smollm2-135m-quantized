"""Quick self-tests. Run:  python test_quantizer.py   (needs only torch)"""
import torch
import torch.nn as nn
import quantizer as Q

torch.manual_seed(0)


def test_pack_roundtrip():
    q = torch.randint(-8, 8, (5, 64), dtype=torch.int8)
    assert torch.equal(Q.unpack4(Q.pack4(q)), q)
    print("ok  pack4/unpack4 round trip")


def test_real_zero_is_exact():
    w = torch.randn(8, 64) * 0.05
    w[:, 3] = 0.0                                  # plant exact zeros
    for bits in (8, 4):
        for gran in ("tensor", "channel", "group"):
            qd, s, z = Q.quantize_weight(w, bits, "asym", gran, 32)
            wd = Q.dequantize_weight(qd, s, z, bits, w.shape)
            assert torch.all(wd[:, 3] == 0), (bits, gran)
    print("ok  real 0 comes back exactly (asymmetric)")


def test_error_is_small_and_ordered():
    w = torch.randn(64, 256) * 0.05
    w[:, 7] *= 40                                  # an outlier column, like in real models
    err = {}
    for gran in ("tensor", "channel", "group"):
        qd, s, z = Q.quantize_weight(w, 8, "asym", gran, 64)
        err[gran] = (w - Q.dequantize_weight(qd, s, z, 8, w.shape)).pow(2).mean().item()
    assert err["group"] <= err["channel"] <= err["tensor"], err
    print("ok  error: tensor %.2e >= channel %.2e >= group %.2e" %
          (err["tensor"], err["channel"], err["group"]))


def test_storage_sizes():
    w = torch.randn(64, 256)
    for bits, expect_bytes in ((8, 64 * 256), (4, 64 * 256 // 2)):
        qd, _, _ = Q.quantize_weight(w, bits, "sym", "channel", 64)
        assert qd.numel() * qd.element_size() == expect_bytes
    print("ok  8-bit = 1 byte/weight, 4-bit = 0.5 byte/weight")


def test_layer_matches_dequantized_matmul():
    lin = nn.Linear(128, 32, bias=True)
    x = torch.randn(4, 128)
    for bits in (8, 4):
        for mode in ("sym", "asym"):
            for gran in ("tensor", "channel", "group"):
                ql = Q.QuantLinear.from_linear(lin, bits, mode, gran, 64)
                ref = torch.nn.functional.linear(x, ql.dequantized_weight(), lin.bias)
                assert torch.allclose(ql(x), ref, atol=1e-5)
    print("ok  QuantLinear forward == matmul with dequantized weights")


def test_sym_uses_the_same_formula():
    """Symmetric = the asymmetric formulas with the range -biggest..+biggest.
    The scale equals biggest / highest_slot and the zero point works out to 0."""
    w = torch.randn(32, 128) * 0.05
    for bits in (8, 4):
        groups = Q.split_into_groups(w, "channel", 64)
        lowest, highest = Q.slot_range(bits, "sym")
        real_min, real_max = Q.real_range(groups, "sym")
        scale = (real_max - real_min) / (highest - lowest)            # the shared scale formula
        assert torch.allclose(scale, groups.abs().amax(1, keepdim=True) / highest)   # = max|x| / qmax
        scale16 = scale.to(torch.float16).float()
        zero_point = torch.round(lowest - real_min / scale16)         # the shared zero point formula
        assert torch.all(zero_point == 0)
    print("ok  symmetric scale = (max - min)/(qmax - qmin) = max|x|/qmax, zero point = 0")


def test_quantize_model_skips_head():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
            self.lm_head = nn.Linear(64, 10)

        def forward(self, x):
            return self.lm_head(self.body(x))

    m = Tiny()
    x = torch.randn(2, 64)
    before = m(x)
    stats = Q.quantize_model(m, bits=8, mode="asym", granularity="channel")
    assert isinstance(m.body[0], Q.QuantLinear) and isinstance(m.lm_head, nn.Linear)
    assert stats["quantized_layers"] == 2 and "lm_head" in stats["skipped"]
    assert (m(x) - before).abs().max() < 0.05
    print("ok  quantize_model swaps layers and skips lm_head")


def test_group_size_check():
    try:
        Q.quantize_weight(torch.randn(4, 576), 4, "asym", "group", 128)
    except ValueError:
        print("ok  group_size 128 on 576 inputs raises a clear error")
        return
    raise AssertionError("should have raised")


if __name__ == "__main__":
    test_pack_roundtrip()
    test_real_zero_is_exact()
    test_error_is_small_and_ordered()
    test_storage_sizes()
    test_sym_uses_the_same_formula()
    test_layer_matches_dequantized_matmul()
    test_quantize_model_skips_head()
    test_group_size_check()
    print("\nall tests passed")
