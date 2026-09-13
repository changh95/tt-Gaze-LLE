# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only (torch, NO device, no ttnn tensors) tests of the ``TT_FUSED=1`` reformulations.

Every exact reformulation the fused device graph relies on is checked here against the
reference math of :mod:`gaze_lle.reference.torch_gaze_lle` in fp32, plus the constant tables,
the host-side per-image work and the knob plumbing. Run with the tree's host python::

    cd models/gaze-lle-p150
    PYTHONPATH=code TT_METAL_HOME=$TREE $TREE/python_env/bin/python -m pytest code/gaze_lle/tests/test_fused_host.py -q

or as a plain script (``python code/gaze_lle/tests/test_fused_host.py``) when pytest is missing.

Tolerances: ``torch.equal`` where the algebra is bit-exact in fp32 (masks, zero-row/gate
identities, constant tables), otherwise max|d| bounds with the reason in the assertion --
fp32 matmul/conv accumulation-order differences are ~1e-6 relative here, far below one bf16 ULP
(~4e-3 relative), which is the precision class of the device path anyway.
"""

from __future__ import annotations

import inspect
import os
import sys

import torch
import torch.nn.functional as F

try:
    import pytest
except ImportError:  # pragma: no cover - plain-script fallback below
    pytest = None

from gaze_lle.reference.torch_gaze_lle import build_gaze_lle
from gaze_lle.tt import fused_math as fm

torch.set_grad_enabled(False)

_REF = None


def _ref():
    """Random-weight ViT-B/14 Gaze-LLE with the in/out head (built once, deterministic)."""
    global _REF
    if _REF is None:
        torch.manual_seed(0)
        _REF = build_gaze_lle("vitb14", inout=True).eval()
    return _REF


def _legacy_im2col(images, n, ps):
    """``TtGazeLLE._reshape_image_for_matmul`` verbatim (the legacy host reshape)."""
    b = images.shape[0]
    return images.view(b, 3, n, ps, n, ps).permute(0, 2, 4, 3, 5, 1).reshape(b, n * n, ps * ps * 3).contiguous()


def _maxdiff(a, b):
    return float((a.float() - b.float()).abs().max())


# --------------------------------------------------------------------------------------
# Knob plumbing
# --------------------------------------------------------------------------------------


def test_knob_default_is_fused_and_zero_selects_legacy_path():
    # Since the device validation (2026-09-13) the fused path is the default; TT_FUSED=0 = legacy.
    cfg = fm.FusedConfig.from_env({})
    assert cfg.enabled is True and fm.select_forward_path(cfg) == "fused"
    for off in ("0", "false", "no", "off"):
        cfg = fm.FusedConfig.from_env({"TT_FUSED": off})
        assert cfg.enabled is False and fm.select_forward_path(cfg) == "legacy", off
    # Sub-knobs cannot turn the path back on once the master switch is off.
    cfg = fm.FusedConfig.from_env({"TT_FUSED": "0", "TT_FUSED_DIT": "1", "TT_FUSED_L1": "1", "TT_FUSED_EAGER": "1"})
    assert cfg.enabled is False and fm.select_forward_path(cfg) == "legacy"
    # An empty/whitespace value is "unset" (= default = fused).
    assert fm.FusedConfig.from_env({"TT_FUSED": " "}).enabled is True
    assert fm.FusedConfig.from_env({"TT_FUSED": "1"}).enabled is True


def test_knob_set_selects_fused_path_and_parses_subknobs():
    cfg = fm.FusedConfig.from_env({"TT_FUSED": "1"})
    assert cfg == fm.FusedConfig.from_env({})  # explicit "1" == the default
    # Validated defaults (DEVICE_VALIDATION.md Results): DIT LoFi, SDPA q128/k128 exact exp, head separate.
    assert cfg.enabled and cfg.trace and cfg.dit and cfg.head == "separate" and cfg.sdpa_pc
    assert cfg.trace_heads == fm.DEFAULT_TRACE_HEADS == (1, 2, 3, 4, 6, 8, 10)
    assert cfg.sdpa_q == 128 and cfg.sdpa_k == 128 and cfg.sdpa_exact_exp
    assert not cfg.minimal_mm and not cfg.l1 and cfg.fidelity == "LoFi" and not cfg.bf16_weights
    assert fm.select_forward_path(cfg) == "fused"
    # The fused layout needs the inout token and a register-free backbone.
    assert fm.select_forward_path(cfg, inout=False) == "legacy"
    assert fm.select_forward_path(cfg, num_register_tokens=4) == "legacy"

    cfg = fm.FusedConfig.from_env({
        "TT_FUSED": "1", "TT_FUSED_EAGER": "1", "TT_FUSED_TRACE_HEADS": "5,1,3,3",
        "TT_FUSED_TRACE_REGION_MB": "90", "TT_FUSED_DIT": "0", "TT_FUSED_HEAD": "separate",
        "TT_FUSED_SDPA": "0", "TT_FUSED_SDPA_Q": "96", "TT_FUSED_SDPA_K": "128",
        "TT_FUSED_SDPA_EXACT_EXP": "0", "TT_FUSED_MINIMAL_MM": "1", "TT_FUSED_L1": "1",
        "TT_FUSED_FIDELITY": "HiFi2", "TT_FUSED_BF16_WEIGHTS": "1",
    })
    assert cfg.trace is False and cfg.trace_heads == (1, 3, 5) and cfg.trace_region_mb == 90
    assert cfg.dit is False and cfg.head == "separate" and cfg.sdpa_pc is False
    assert cfg.sdpa_q == 96 and cfg.sdpa_k == 128 and cfg.sdpa_exact_exp is False
    assert cfg.minimal_mm and cfg.l1 and cfg.fidelity == "HiFi2" and cfg.bf16_weights
    assert cfg.as_dict()["trace_heads"] == [1, 3, 5]

    for bad in ({"TT_FUSED_HEAD": "nope"}, {"TT_FUSED_FIDELITY": "HiFi9"}, {"TT_FUSED_SDPA_Q": "48"},
                {"TT_FUSED_TRACE_HEADS": "0,1"}):
        try:
            fm.FusedConfig.from_env({"TT_FUSED": "1", **bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_wrapper_dispatch_and_open_device_kwargs():
    """The ttnn-importing wrapper consults the knob (import only: no device, no ttnn tensors)."""
    try:
        from gaze_lle.tt import tt_gaze_lle as m
    except ImportError as e:  # ttnn not importable in this python: the torch-only tests still count
        if pytest is not None:
            pytest.skip(f"host ttnn not importable here: {e}")
        print(f"SKIP wrapper dispatch check: {e}")
        return
    assert m.open_device_kwargs(fm.FusedConfig.from_env({"TT_FUSED": "0"})) == {}  # legacy: bare open_device
    assert m.open_device_kwargs(fm.FusedConfig.from_env({})) == {"trace_region_size": 64 << 20}  # default = fused
    assert m.open_device_kwargs(fm.FusedConfig.from_env({"TT_FUSED": "1"})) == {"trace_region_size": 64 << 20}
    assert m.open_device_kwargs(fm.FusedConfig.from_env({"TT_FUSED": "1", "TT_FUSED_EAGER": "1"})) == {}
    assert m.open_device_kwargs(fm.FusedConfig.from_env({"TT_FUSED": "1", "TT_FUSED_TRACE_REGION_MB": "8"})) == {
        "trace_region_size": 8 << 20}
    # __init__ reads the knob once and dispatches; __call__ dispatches on the stored flag.
    src_init = inspect.getsource(m.TtGazeLLE.__init__)
    assert "FusedConfig.from_env()" in src_init and "select_forward_path(" in src_init
    assert "self._init_fused(ref_model, device)" in src_init
    assert "if self.fused:" in inspect.getsource(m.TtGazeLLE.__call__)
    assert m._FusedMixin in m.TtGazeLLE.__mro__
    for name in ("warmup", "release_traces", "fused_info", "_run_traced", "_run_eager",
                 "_fused_scene_graph", "_fused_decoder_graph", "_ensure_scene_trace", "_ensure_decoder_trace"):
        assert callable(getattr(m.TtGazeLLE, name)), name
    # Legacy methods are still there, untouched in name.
    for name in ("_encode_scene", "_build_head_contrib", "_decode_head", "_reshape_image_for_matmul"):
        assert hasattr(m.TtGazeLLE, name), name


# --------------------------------------------------------------------------------------
# Host per-image work
# --------------------------------------------------------------------------------------


def test_im2col_fill_matches_legacy_reshape_and_leaves_pad_zero():
    ref = _ref()
    n, ps = ref.backbone.img_size // ref.backbone.patch_size, ref.backbone.patch_size
    k = ps * ps * 3
    torch.manual_seed(1)
    images = torch.randn(1, 3, 448, 448)
    legacy = _legacy_im2col(images, n, ps)  # (1, 1024, 588) fp32

    buf = torch.zeros(1, n * n + 1, fm.pad_to_tile(k), dtype=torch.float32)
    fm.fill_im2col(buf, images, n, ps)
    assert buf.shape == (1, 1025, 608)
    assert torch.equal(buf[:, : n * n, :k], legacy)
    assert torch.equal(buf[:, n * n], torch.zeros(1, 608)), "CLS row must stay zero"
    assert torch.equal(buf[:, :, k:], torch.zeros(1, 1025, 608 - k)), "pad columns must stay zero"

    # The upload buffer is bf16: the cast is torch's round-to-nearest-even, same as .to(bfloat16).
    buf16 = torch.zeros(1, n * n + 1, fm.pad_to_tile(k), dtype=torch.bfloat16)
    fm.fill_im2col(buf16, images, n, ps)
    assert torch.equal(buf16[:, : n * n, :k], legacy.to(torch.bfloat16))
    # Second fill overwrites completely (persistent buffer reuse).
    fm.fill_im2col(buf16, -images, n, ps)
    assert torch.equal(buf16[:, : n * n, :k], (-legacy).to(torch.bfloat16))
    try:
        fm.fill_im2col(buf16, torch.zeros(2, 3, 448, 448), n, ps)
    except ValueError:
        pass
    else:
        raise AssertionError("B != 1 must be rejected")


def test_head_mask_matches_reference_bbox_to_head_map():
    ref = _ref()
    fh, fw = ref.featmap_h, ref.featmap_w
    bboxes = [
        (0.3, 0.2, 0.6, 0.5),          # the warm-up bbox
        (0.0, 0.0, 1.0, 1.0),          # whole image
        (0.38, 0.10, 0.53, 0.43),      # test_pretrained_eval case 1
        (0.40, 0.15, 0.60, 0.60),      # case 2
        (0.015625, 0.015625, 0.046875, 0.046875),  # ties at .5: python round() is banker's
        (0.0, 0.0, 0.01, 0.01),        # rounds to an empty box (all-zero mask, as the reference)
        (0.98, 0.97, 1.0, 1.0),        # bottom-right corner
        (0.232, 0.083832, 0.352, 0.311377),  # smoke_test bbox 0 on source_1.png
    ]
    masks = fm.build_head_mask(bboxes, fh, fw)  # (N, 1025, 1)
    assert masks.shape == (len(bboxes), fh * fw + 1, 1)
    idx_h = torch.arange(fh).view(1, fh, 1).float()
    idx_w = torch.arange(fw).view(1, 1, fw).float()
    for i, bb in enumerate(bboxes):
        expected = ref._bbox_to_head_map(bb)  # (32, 32)
        got = masks[i, : fh * fw, 0].reshape(fh, fw)
        assert torch.equal(got, expected), f"mask {i} != reference _bbox_to_head_map for {bb}"
        assert float(masks[i, fh * fw, 0]) == 0.0, "special-token row must be 0"
        # ... and equals the legacy device cascade (ge/lt/mul on index grids, same rounding).
        x0, y0, x1, y1 = round(bb[0] * fw), round(bb[1] * fh), round(bb[2] * fw), round(bb[3] * fh)
        cascade = ((idx_h >= y0).float() * (idx_h < y1).float()) * ((idx_w >= x0).float() * (idx_w < x1).float())
        assert torch.equal(got, cascade.reshape(fh, fw))
    assert masks[0].sum() == (round(0.6 * fw) - round(0.3 * fw)) * (round(0.5 * fh) - round(0.2 * fh))
    assert masks[5].sum() == 0


def test_bucket_selection_and_padding():
    buckets = (1, 2, 3, 4, 6, 8, 10)
    assert fm.pick_bucket(1, buckets) == 1
    assert fm.pick_bucket(3, buckets) == 3
    assert fm.pick_bucket(5, buckets) == 6
    assert fm.pick_bucket(7, buckets) == 8
    assert fm.pick_bucket(10, buckets) == 10
    assert fm.pick_bucket(11, buckets) is None
    assert fm.pick_bucket(2, ()) is None
    bb = [(0.1, 0.1, 0.2, 0.2), (0.5, 0.5, 0.9, 0.9)]
    padded = fm.pad_bboxes(bb, 6)
    assert len(padded) == 6 and padded[:2] == bb and all(p == bb[-1] for p in padded[2:])
    assert fm.pad_bboxes(bb, 2) == bb
    try:
        fm.pad_bboxes(bb, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("bucket smaller than N must be rejected")


def test_decode_head_output_layout():
    fh, fw = 32, 32
    n_bucket, n_real = 4, 3
    torch.manual_seed(2)
    out = torch.rand(n_bucket, fh * fw + 1, 5)
    heat, inout = fm.decode_head_output(out, fh, fw, n_real, (64, 64))
    assert heat.shape == (n_real, 64, 64) and inout.shape == (n_real,)
    assert torch.equal(inout, out[:n_real, fh * fw, 4])
    # Same fold as the legacy (N, 1024, 4) -> (N, 32, 32, 2, 2) -> permute(0,1,3,2,4) -> (N, 64, 64).
    legacy = out[:n_real, : fh * fw, :4].reshape(n_real, fh, fw, 2, 2).permute(0, 1, 3, 2, 4).reshape(n_real, 64, 64)
    assert torch.equal(heat, legacy)
    for n, i, j, a, b in [(0, 0, 0, 0, 0), (2, 31, 31, 1, 1), (1, 5, 17, 1, 0), (0, 12, 3, 0, 1)]:
        assert heat[n, 2 * i + a, 2 * j + b] == out[n, i * fw + j, 2 * a + b]
    # Legacy-head fallback decode (two outputs) agrees.
    heat2, inout2 = fm.decode_legacy_head_outputs(out[..., :4], out[..., 4:5], fh, fw, n_real, (64, 64))
    assert torch.equal(heat2, heat) and torch.equal(inout2, inout)
    # A different out_size goes through the same bilinear resize as the legacy path.
    heat3, _ = fm.decode_head_output(out, fh, fw, n_real, (48, 48))
    assert heat3.shape == (n_real, 48, 48)


# --------------------------------------------------------------------------------------
# Exact reformulations (constant tables vs reference math)
# --------------------------------------------------------------------------------------


def test_patch_embed_fused_matches_reference_embedding():
    """X_pad @ W_pad + C_patch == fused-order [conv(patches) + pos_patch ; cls + pos_cls]."""
    ref = _ref()
    bb = ref.backbone
    n, ps = bb.img_size // bb.patch_size, bb.patch_size
    k = ps * ps * 3
    torch.manual_seed(3)
    images = torch.randn(1, 3, 448, 448)

    w_pad = fm.build_patch_weight(bb)
    c_patch = fm.build_patch_const(bb)
    assert w_pad.shape == (608, 768) and c_patch.shape == (1, 1025, 768)
    assert torch.equal(w_pad[k:], torch.zeros(608 - k, 768))
    # W_pad rows are exactly the legacy flattening of the conv kernel.
    legacy_w = bb.patch_embed_proj.weight.detach().permute(2, 3, 1, 0).reshape(-1, 768)
    assert torch.equal(w_pad[:k], legacy_w)

    x_pad = torch.zeros(1, 1025, 608)
    fm.fill_im2col(x_pad, images, n, ps)
    got = x_pad @ w_pad + c_patch  # (1, 1025, 768), CLS at row 1024

    # Reference: conv -> flatten -> cat([cls, patches]) + pos_embed (DinoV2Backbone.forward).
    patches = bb.patch_embed_proj(images).flatten(2).transpose(1, 2)
    x_ref = torch.cat([bb.cls_token, patches], dim=1) + bb.pos_embed
    expected = fm.legacy_to_fused_order(x_ref)
    d = _maxdiff(got, expected)
    assert d <= 1e-4, f"patch-embed max|d| {d} (fp32 conv vs matmul accumulation order only)"
    # The CLS row is bit-exact: its im2col row is zero, so 0 @ W == 0 and the constant passes through.
    assert torch.equal(got[:, 1024], (bb.cls_token + bb.pos_embed[:, :1]).reshape(1, 768))
    # ... and the pad columns contribute exact zeros (padded and unpadded matmul agree bitwise).
    assert torch.equal(x_pad[:, :1024, :k] @ w_pad[:k], x_pad[:, :1024] @ w_pad)


def test_blocks_are_equivariant_to_moving_the_special_token_last():
    """block(P x) == P block(x) for the DINOv2 block, the final LN and the gaze block."""
    ref = _ref()
    torch.manual_seed(4)
    x = torch.randn(1, 1025, 768)
    for i in (0, 11):
        blk = ref.backbone.blocks[i]
        a = blk(fm.legacy_to_fused_order(x))
        b = fm.legacy_to_fused_order(blk(x))
        d = _maxdiff(a, b)
        assert d <= 1e-4, f"block {i} not equivariant: max|d| {d} (fp32 softmax/matmul order)"
    a = ref.backbone.norm(fm.legacy_to_fused_order(x))
    b = fm.legacy_to_fused_order(ref.backbone.norm(x))
    assert torch.equal(a, b), "LayerNorm is row-wise: bit-exact under a row permutation"

    torch.manual_seed(5)
    y = torch.randn(3, 1025, 256)  # N=3 heads batched
    for gb in ref.transformer:
        a = gb(fm.legacy_to_fused_order(y))
        b = fm.legacy_to_fused_order(gb(y))
        d = _maxdiff(a, b)
        assert d <= 1e-4, f"gaze block not equivariant: max|d| {d}"
    # Batched heads are independent: per-head results equal single-head runs (bit-exact per row).
    gb = ref.transformer[0]
    batched = gb(y)
    for j in range(3):
        d = _maxdiff(batched[j : j + 1], gb(y[j : j + 1]))
        assert d <= 1e-5, f"batch element {j} differs from its single run by {d}"
    # And the two orders are inverse of each other.
    assert torch.equal(fm.fused_to_legacy_order(fm.legacy_to_fused_order(x)), x)


def test_gated_projection_builds_decoder_base_exactly():
    """C2 + (x_final @ W_proj) * G == [proj(x_patches) + bias + gaze_pos ; inout_token]."""
    ref = _ref()
    torch.manual_seed(6)
    x_final = torch.randn(1, 1025, 768)
    x_final[:, 1024] = 1e3 * torch.randn(768)  # garbage in the CLS row must not leak

    c2, g = fm.build_gated_proj_consts(ref)
    w_proj = fm.build_proj_weight(ref)
    assert c2.shape == g.shape == (1, 1025, 256) and w_proj.shape == (768, 256)
    assert torch.equal(g[:, :1024], torch.ones(1, 1024, 256)) and torch.equal(g[:, 1024], torch.zeros(1, 256))
    x_base = c2 + (x_final @ w_proj) * g

    # Reference: GazeLLE.forward -> linear (1x1 conv) + pos_embed, then cat([inout_tok, x]).
    feat = x_final[:, :1024].reshape(1, 32, 32, 768).permute(0, 3, 1, 2)
    x_ref = ref.linear(feat) + ref.pos_embed.unsqueeze(0)  # (1, 256, 32, 32)
    x_ref = x_ref.flatten(2).permute(0, 2, 1)  # (1, 1024, 256)
    d = _maxdiff(x_base[:, :1024], x_ref)
    assert d <= 1e-4, f"gated proj patch rows max|d| {d} (bias+pos folded once; fp32 conv vs matmul)"
    assert torch.equal(x_base[:, 1024], ref.inout_token.weight.detach().reshape(1, 256)), \
        "row 1024 must be the inout token bit-exactly (0 * finite == 0)"
    # Legacy-order equivalence with the reference decoder input (torch.cat([inout_tok, x])).
    legacy = torch.cat([ref.inout_token.weight.unsqueeze(0), x_ref], dim=1)
    assert _maxdiff(fm.fused_to_legacy_order(x_base), legacy) <= 1e-4


def test_head_conditioning_matches_reference():
    """X_base + M' * head_token == reference head_map * head_token add (patch rows), row 1024 untouched."""
    ref = _ref()
    torch.manual_seed(7)
    x_base = torch.randn(1, 1025, 256)
    bboxes = [(0.3, 0.2, 0.6, 0.5), (0.0, 0.0, 1.0, 1.0)]
    mask = fm.build_head_mask(bboxes, 32, 32)
    x0 = x_base + mask * ref.head_token.weight.detach().reshape(1, 1, 256)  # (2, 1025, 256)
    for i, bb in enumerate(bboxes):
        hm = ref._bbox_to_head_map(bb)  # (32, 32)
        contrib = (hm.unsqueeze(0) * ref.head_token.weight.detach().unsqueeze(-1).unsqueeze(-1))  # (1, 256, 32, 32)
        expected = x_base[:, :1024] + contrib.flatten(2).permute(0, 2, 1)
        assert torch.equal(x0[i : i + 1, :1024], expected)
        assert torch.equal(x0[i, 1024], x_base[0, 1024])


def test_fused_heatmap_head_matches_conv_modules():
    """(256, 4) fused weight + scalar bias + decode fold == ConvTranspose2d(2,2) -> Conv2d(1x1) -> Sigmoid."""
    ref = _ref()
    w_hm, b_hm = fm.build_heatmap_head(ref)
    assert w_hm.shape == (256, 4)
    # Same formula as the legacy TtGazeLLE constructor.
    ct_w = ref.heatmap_head[0].weight.detach()
    ct_b = ref.heatmap_head[0].bias.detach()
    c1_w = ref.heatmap_head[1].weight.detach().squeeze(-1).squeeze(-1)
    legacy_w = torch.einsum("ko,ioab->ikab", c1_w, ct_w).squeeze(1).reshape(256, 4)
    assert torch.equal(w_hm, legacy_w) and b_hm == float((c1_w @ ct_b).squeeze().item())

    torch.manual_seed(8)
    x = torch.randn(2, 1024, 256)  # decoder output, patch rows
    got = torch.sigmoid(x @ w_hm + b_hm)  # (2, 1024, 4)
    heat, _ = fm.decode_head_output(F.pad(got, (0, 1, 0, 1)), 32, 32, 2, (64, 64))
    feat = x.reshape(2, 32, 32, 256).permute(0, 3, 1, 2)
    expected = ref.heatmap_head(feat).squeeze(1)  # (2, 64, 64)
    d = _maxdiff(heat, expected)
    assert d <= 1e-5, f"fused heatmap head vs conv modules max|d| {d}"


def test_block_diagonal_head_is_exact():
    """sigmoid([x, h] @ W_head + b) reproduces the heatmap head (cols 0..3) and the in/out head (col 4)."""
    ref = _ref()
    w_head, b_head = fm.build_fused_head(ref)
    w_hm, b_hm = fm.build_heatmap_head(ref)
    w1, b1, w2, b2 = fm.build_inout_head(ref)
    assert w_head.shape == (384, 5) and b_head.shape == (1, 5)
    assert torch.equal(w_head[:256, :4], w_hm) and torch.equal(w_head[256:, 4], w2[:, 0])
    assert torch.equal(w_head[256:, :4], torch.zeros(128, 4)) and torch.equal(w_head[:256, 4], torch.zeros(256))
    assert torch.equal(b_head[0, :4], torch.full((4,), b_hm)) and b_head[0, 4] == b2[0, 0]
    # K-concat seam: prefix K 256 and suffix K 128 are tile multiples and sum to the weight K.
    assert 256 % fm.TILE == 0 and 128 % fm.TILE == 0 and fm.pad_to_tile(256) + fm.pad_to_tile(128) == w_head.shape[0]

    torch.manual_seed(9)
    x = torch.randn(3, 1025, 256)
    h = F.relu(x @ w1 + b1)
    out = torch.sigmoid(torch.cat([x, h], dim=-1) @ w_head + b_head)  # (3, 1025, 5)
    heat_ref = torch.sigmoid(x @ w_hm + b_hm)
    d = _maxdiff(out[..., :4], heat_ref)
    assert d <= 1e-6, f"heatmap columns max|d| {d} (zero block adds exact zeros; fp32 K-blocking only)"
    # In/out: the reference module chain on the special-token row.
    io_ref = ref.inout_head(x[:, 1024, :]).squeeze(-1)  # Linear, ReLU, Dropout(eval), Linear, Sigmoid
    d = _maxdiff(out[:, 1024, 4], io_ref)
    assert d <= 1e-6, f"inout column max|d| {d}"
    # h on the patch rows is computed but never read: nothing else depends on it.


def test_torch_fused_forward_matches_reference_per_head():
    """The whole fused layout in fp32 (same constants as the device path) vs N single-person reference forwards."""
    ref = _ref()
    torch.manual_seed(10)
    images = torch.randn(1, 3, 448, 448)
    bboxes = [(0.3, 0.2, 0.6, 0.5), (0.38, 0.10, 0.53, 0.43), (0.0, 0.0, 1.0, 1.0)]
    cap = {}
    fused = fm.torch_fused_forward(ref, images, bboxes, capture=cap)
    assert fused["heatmap"].shape == (3, 64, 64) and fused["inout"].shape == (3,)
    assert cap["after_prefix"].shape == (1, 1025, 768) and cap["head_out"].shape == (3, 1025, 5)
    worst_hm = worst_io = 0.0
    for i, bb in enumerate(bboxes):
        expected = ref(images, [bb])
        worst_hm = max(worst_hm, _maxdiff(fused["heatmap"][i], expected["heatmap"][0]))
        worst_io = max(worst_io, abs(float(fused["inout"][i]) - float(expected["inout"][0])))
    print(f"\nfused-vs-reference fp32: heatmap max|d| {worst_hm:.3e}, inout max|d| {worst_io:.3e}")
    assert worst_hm <= 1e-4, f"heatmap max|d| {worst_hm} > 1e-4 (fp32 accumulation-order class only)"
    assert worst_io <= 1e-5, f"inout max|d| {worst_io} > 1e-5"


# --------------------------------------------------------------------------------------
# Shape arithmetic the device ops validate (computed, not built as ttnn tensors)
# --------------------------------------------------------------------------------------


def test_padded_shape_arithmetic_for_device_ops():
    ref = _ref()
    S = ref.backbone.num_patches + 1
    assert S == 1025 and fm.pad_to_tile(S) == 1056
    k = ref.backbone.patch_size ** 2 * 3
    assert k == 588 and fm.pad_to_tile(k) == 608
    # RM upload rows must be 32-byte aligned pages: 608 bf16 = 1216 B = 38 x 32.
    assert (608 * 2) % 32 == 0
    # minimal_matmul K == K_w for the fused patch embed (logical 608 both sides).
    assert fm.build_patch_weight(ref.backbone).shape[0] == 608
    # dit fused ternary_a [M, N] == output; ternary_b [1, N] (ones) or [M, N] (gate).
    assert fm.build_patch_const(ref.backbone).shape == (1, S, 768)
    c2, g = fm.build_gated_proj_consts(ref)
    assert c2.shape == (1, S, 256) and g.shape == (1, S, 256)
    # Head matmul: bias last dim N=5 (padded 32), weight K 384 == 256 + 128 padded, output (N, S, 5).
    w, b = fm.build_fused_head(ref)
    assert w.shape == (384, 5) and b.shape == (1, 5) and fm.pad_to_tile(5) == 32
    # SDPA program-config chunk sizes are tile multiples; S 1056 padded to the chunk: 1056 -> 1152 (q/k 128).
    cfg = fm.FusedConfig.from_env({"TT_FUSED": "1"})
    assert cfg.sdpa_q % 32 == 0 and cfg.sdpa_k % 32 == 0
    assert -(-1056 // cfg.sdpa_k) * cfg.sdpa_k == 1152 and -(-1056 // cfg.sdpa_q) * cfg.sdpa_q == 1152
    # Per-bucket persistent buffers: mask (n, 1056, 32) bf16 and output (n, 1056, 32) bf16 = 66 KiB each per head.
    assert 1056 * 32 * 2 == 67584
    # Op counts of the fused graph (any N) vs legacy 201 / 315 / 714 at N = 1 / 3 / 10.
    scene = 1 + 1 + 12 * 9 + 1 + 1
    decoder = 2 + 3 * 9 + 2
    assert scene == 112 and decoder == 31 and scene + decoder == 143
    legacy = lambda n: 144 + 57 * n  # noqa: E731
    assert (legacy(1), legacy(3), legacy(10)) == (201, 315, 714)


if __name__ == "__main__":  # plain-script fallback when pytest is unavailable
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{failures} failure(s)")
    sys.exit(1 if failures else 0)
