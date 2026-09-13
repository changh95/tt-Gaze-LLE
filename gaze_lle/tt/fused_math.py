# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Torch-only half of the ``TT_FUSED=1`` Gaze-LLE pipeline (no ttnn import).

Everything the fused device graph needs that is *not* a device op lives here so it can be
tested on a host without a chip (``tests/test_fused_host.py``):

* the ``TT_FUSED*`` knob parsing (:class:`FusedConfig`, read ONCE at model build),
* the exact reformulations as constant tables built from the reference ``nn.Module``s,
* the host-side per-image work (im2col fill, head mask, output decode),
* a pure-torch fp32 mirror of the fused device graph (:func:`torch_fused_forward`) that the
  tests compare against the reference :class:`GazeLLE` forward.

Token layout of the fused graph (differs from the legacy path and from the torch reference):
the sequence is ``[patch_0 .. patch_1023, CLS]`` in the backbone and ``[patch_0 .. patch_1023,
inout]`` in the gaze decoder, i.e. the special token sits at row ``num_patches`` (1024) instead
of row 0. Transformer blocks are permutation-equivariant (LayerNorm / MLP are row-wise and
attention has no positional term after the pos-embed add), so this is exact math; it makes the
special-token row the LAST row of the last tile, which is what turns the legacy 4-op tile-padded
``concat`` fallbacks and 3-op unaligned ``slice`` fallbacks into zero ops:

* patch embedding: ``X_pad @ W_pad + C_patch`` where ``X_pad`` (1, 1025, 608) is the bf16 im2col
  with a zero row 1024 (CLS slot) and zero columns 588..607, ``W_pad`` (608, 768) zero-extended,
  and ``C_patch`` rows 0..1023 = ``pos_patch + conv_bias``, row 1024 = ``cls + pos_cls``;
* gated projection: ``X_base = C2 + (x_final @ W_proj) * G`` with ``G`` = 1 on patch rows and 0 on
  row 1024 (``0 * finite = 0`` exactly, so the CLS row's garbage projection is dropped) and
  ``C2`` rows 0..1023 = ``gaze_pos + proj_bias``, row 1024 = ``inout_token``;
* head conditioning: ``X0 = X_base + M' * head_token`` with a host-built ``M'`` (N, 1025, 1)
  equal to the reference ``_bbox_to_head_map`` on rows 0..1023 and 0 on row 1024;
* heads: ``h = relu(X @ W1 + b1)`` on every row, then ``sigmoid([X, h] @ W_head + b_head)`` with the
  block-diagonal ``W_head`` (384, 5) = ``[[W_heatmap, 0], [0, W_inout2]]`` so ONE readback
  (N, 1025, 5) carries the heatmap (rows 0..1023, cols 0..3) and the in/out score (row 1024, col 4).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

TILE = 32

DEFAULT_TRACE_HEADS = (1, 2, 3, 4, 6, 8, 10)
DEFAULT_TRACE_REGION_MB = 64


def pad_to_tile(n: int) -> int:
    """Smallest multiple of the 32-wide tile that is >= ``n``."""
    return -(-int(n) // TILE) * TILE


def _env_flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    v = env.get(name)
    if v is None or not str(v).strip():
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    v = env.get(name)
    if v is None or not str(v).strip():
        return default
    return int(v)


@dataclass(frozen=True)
class FusedConfig:
    """All ``TT_FUSED*`` knobs, read once at model build (``FusedConfig.from_env()``).

    ``enabled`` is the master switch. Since the device validation of 2026-09-13 the fused path is
    the DEFAULT (``TT_FUSED`` unset or ``1``); ``TT_FUSED=0`` selects the untouched legacy path, in
    which case every other field is irrelevant. The sub-knobs exist so each fusion can be A/B'd
    against its legacy-op fallback inside the fused layout; the defaults below are the validated
    configuration (DEVICE_VALIDATION.md "Results").
    """

    enabled: bool = True
    # Whole-graph metal trace (scene trace + one decoder trace per head-count bucket).
    # TT_FUSED_EAGER=1 runs the same fused graph eagerly (debug / A/B of the trace itself).
    trace: bool = True
    trace_heads: Tuple[int, ...] = DEFAULT_TRACE_HEADS
    trace_region_mb: int = DEFAULT_TRACE_REGION_MB
    # dit_minimal_matmul_addcmul_fused for patch-embed, proj+residual, fc2+residual, gated proj.
    # Off: the same layout with ttnn.linear + add/mul (bfp8 proj/fc2 weights like legacy).
    dit: bool = True
    # "separate" (default, validated): minimal_matmul K-concat + separate ttnn.sigmoid (3 ops, 1 readback)
    #   -- measured 0.0 / -0.05 / -0.15 ms vs "fused" at N = 1 / 3 / 10 (the fused SFPU pass runs on the
    #   whole padded output), same sigmoid kernel as legacy;
    # "fused": the same matmul with the sigmoid fused in (2 ops, 1 readback);
    # "legacy": in/out MLP + heatmap linear/add/sigmoid on all rows (6 ops, 2 readbacks).
    head: str = "separate"
    # SDPAProgramConfig (chunk sizes + exact exp) instead of the default 32/32 chunks, exp_approx on.
    # q128/k128 measured on the p150a: backbone SDPA 0.227 -> 0.098 ms (q64/k256: 0.124), whole forward
    # -0.22 / -0.23 / -0.31 ms vs q64/k256 at N = 1 / 3 / 10, GazeFollow-500 L2 0.1143 / 0.0503 (best).
    sdpa_pc: bool = True
    sdpa_q: int = 128
    sdpa_k: int = 128
    sdpa_exact_exp: bool = True
    # minimal_matmul for qkv / fc1 (+ separate exact gelu) instead of ttnn.linear (unverified on HW).
    minimal_mm: bool = False
    # Backbone working set in L1 (unverified fit for 768/3072-wide activations).
    l1: bool = False
    # Math fidelity of the fused matmuls (dit fused / minimal_matmul). Legacy runs LoFi everywhere.
    fidelity: str = "LoFi"
    # bf16 instead of bfp8 for the qkv / fc1 backbone weights (+70 MB DRAM).
    bf16_weights: bool = False

    @staticmethod
    def from_env(env: Optional[Mapping[str, str]] = None) -> "FusedConfig":
        env = os.environ if env is None else env
        enabled = _env_flag(env, "TT_FUSED", True)
        heads_raw = env.get("TT_FUSED_TRACE_HEADS") or ""
        heads = tuple(sorted({int(t) for t in heads_raw.replace(";", ",").split(",") if t.strip()})) \
            if heads_raw.strip() else DEFAULT_TRACE_HEADS
        if any(h < 1 for h in heads):
            raise ValueError(f"TT_FUSED_TRACE_HEADS must be positive integers, got {heads_raw!r}")
        head = (env.get("TT_FUSED_HEAD") or "separate").strip().lower()
        if head not in ("fused", "separate", "legacy"):
            raise ValueError(f"TT_FUSED_HEAD must be fused|separate|legacy, got {head!r}")
        fidelity = (env.get("TT_FUSED_FIDELITY") or "LoFi").strip()
        if fidelity not in ("LoFi", "HiFi2", "HiFi3", "HiFi4"):
            raise ValueError(f"TT_FUSED_FIDELITY must be LoFi|HiFi2|HiFi3|HiFi4, got {fidelity!r}")
        sdpa_q = _env_int(env, "TT_FUSED_SDPA_Q", 128)
        sdpa_k = _env_int(env, "TT_FUSED_SDPA_K", 128)
        if sdpa_q % TILE or sdpa_k % TILE or sdpa_q <= 0 or sdpa_k <= 0:
            raise ValueError(f"SDPA chunk sizes must be positive multiples of 32, got q={sdpa_q} k={sdpa_k}")
        return FusedConfig(
            enabled=enabled,
            trace=not _env_flag(env, "TT_FUSED_EAGER", False),
            trace_heads=heads,
            trace_region_mb=max(1, _env_int(env, "TT_FUSED_TRACE_REGION_MB", DEFAULT_TRACE_REGION_MB)),
            dit=_env_flag(env, "TT_FUSED_DIT", True),
            head=head,
            sdpa_pc=_env_flag(env, "TT_FUSED_SDPA", True),
            sdpa_q=sdpa_q,
            sdpa_k=sdpa_k,
            sdpa_exact_exp=_env_flag(env, "TT_FUSED_SDPA_EXACT_EXP", True),
            minimal_mm=_env_flag(env, "TT_FUSED_MINIMAL_MM", False),
            l1=_env_flag(env, "TT_FUSED_L1", False),
            fidelity=fidelity,
            bf16_weights=_env_flag(env, "TT_FUSED_BF16_WEIGHTS", False),
        )

    def as_dict(self) -> Dict[str, object]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def select_forward_path(cfg: FusedConfig, *, inout: bool = True, num_register_tokens: int = 0) -> str:
    """``"fused"`` or ``"legacy"``: the path :class:`TtGazeLLE` builds for this config.

    The fused layout hard-codes the ``[patches, special-token]`` sequence with exactly one special
    token in each stage, so it needs ``inout=True`` and a register-free backbone (the served
    ``vitb14`` checkpoint has 0 register tokens); anything else falls back to the legacy path.
    """
    if not cfg.enabled:
        return "legacy"
    if not inout or num_register_tokens != 0:
        return "legacy"
    return "fused"


# --------------------------------------------------------------------------------------
# Per-image host work
# --------------------------------------------------------------------------------------


def im2col_patches(images: torch.Tensor, num_patches_side: int, patch_size: int) -> torch.Tensor:
    """(B, 3, H, W) -> (B, num_patches, ps*ps*3) view/permute/reshape (the legacy host reshape).

    Feature order is ``(kh, kw, c)`` -- the same flattening as ``build_patch_weight``.
    """
    b = images.shape[0]
    n, ps = num_patches_side, patch_size
    return images.view(b, 3, n, ps, n, ps).permute(0, 2, 4, 3, 5, 1).reshape(b, n * n, ps * ps * 3)


def fill_im2col(buf: torch.Tensor, images: torch.Tensor, num_patches_side: int, patch_size: int) -> torch.Tensor:
    """One strided copy of the im2col into the persistent upload buffer ``buf`` (1, S, K_pad).

    ``buf`` rows 0..num_patches-1 / cols 0..ps*ps*3-1 receive the patches (cast to ``buf.dtype``,
    i.e. torch's round-to-nearest-even when ``buf`` is bf16); the CLS row ``num_patches`` and the
    pad columns stay zero (they are never written). Returns ``buf``.
    """
    if images.shape[0] != 1:
        raise ValueError("the fused device graph is built for one image per forward (B=1)")
    n, ps = num_patches_side, patch_size
    k = ps * ps * 3
    buf[:, : n * n, :k].copy_(im2col_patches(images, n, ps))
    return buf


def build_head_mask(bboxes: Sequence[Sequence[float]], featmap_h: int, featmap_w: int,
                    seq_len: Optional[int] = None) -> torch.Tensor:
    """Host-built head masks (N, seq_len, 1) fp32; row ``featmap_h*featmap_w`` (the special token) is 0.

    Rows 0..h*w-1 equal the reference ``GazeLLE._bbox_to_head_map(bbox).flatten()`` bit for bit:
    the same python ``round()`` (ties to even) of ``coord * grid`` and the same half-open slice.
    """
    h, w = featmap_h, featmap_w
    seq_len = (h * w + 1) if seq_len is None else int(seq_len)
    if seq_len < h * w:
        raise ValueError("seq_len must hold all patch rows")
    out = torch.zeros(len(bboxes), seq_len, 1, dtype=torch.float32)
    for i, bb in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bb
        x0, y0, x1, y1 = round(xmin * w), round(ymin * h), round(xmax * w), round(ymax * h)
        m = torch.zeros(h, w, dtype=torch.float32)
        m[y0:y1, x0:x1] = 1.0
        out[i, : h * w, 0] = m.reshape(-1)
    return out


def pick_bucket(n: int, buckets: Sequence[int]) -> Optional[int]:
    """Smallest captured head-count bucket >= n (None when n exceeds every bucket)."""
    fits = [b for b in buckets if b >= n]
    return min(fits) if fits else None


def pad_bboxes(bboxes: Sequence[Sequence[float]], n_bucket: int) -> List[Sequence[float]]:
    """Repeat the last bbox up to the bucket size (dummy heads; their outputs are dropped)."""
    if n_bucket < len(bboxes):
        raise ValueError(f"bucket {n_bucket} smaller than {len(bboxes)} heads")
    out = list(bboxes)
    out.extend([bboxes[-1]] * (n_bucket - len(bboxes)))
    return out


def decode_head_output(out: torch.Tensor, featmap_h: int, featmap_w: int, n_real: int,
                       out_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """(N_bucket, S, 5) fused head output -> (heatmap (n_real, 2h, 2w), inout (n_real,)).

    Columns 0..3 of the patch rows hold the fused ConvTranspose2d(2,2)+Conv2d(1x1) output for the
    2x2 sub-pixels of each patch (same ``(h, w, 2, 2) -> (2h, 2w)`` fold as the legacy path);
    row ``h*w`` column 4 holds the in/out probability.
    """
    h, w = featmap_h, featmap_w
    out = out[:n_real].float()
    hm = out[:, : h * w, :4].reshape(n_real, h, w, 2, 2).permute(0, 1, 3, 2, 4).reshape(n_real, 2 * h, 2 * w)
    if out_size is not None and tuple(out_size) != (2 * h, 2 * w):
        hm = F.interpolate(hm.unsqueeze(1), size=tuple(out_size), mode="bilinear", align_corners=False).squeeze(1)
    inout = out[:, h * w, 4].reshape(n_real)
    return hm, inout


def decode_legacy_head_outputs(hm: torch.Tensor, io: torch.Tensor, featmap_h: int, featmap_w: int, n_real: int,
                               out_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """``TT_FUSED_HEAD=legacy`` decode: hm (N, S, 4) rows 0..h*w-1 and io (N, S, 1) row h*w."""
    h, w = featmap_h, featmap_w
    hm = hm[:n_real].float()
    heat = hm[:, : h * w, :4].reshape(n_real, h, w, 2, 2).permute(0, 1, 3, 2, 4).reshape(n_real, 2 * h, 2 * w)
    if out_size is not None and tuple(out_size) != (2 * h, 2 * w):
        heat = F.interpolate(heat.unsqueeze(1), size=tuple(out_size), mode="bilinear", align_corners=False).squeeze(1)
    inout = io[:n_real].float()[:, h * w, 0].reshape(n_real)
    return heat, inout


# --------------------------------------------------------------------------------------
# Constant tables (fp32, built from the reference modules so they cannot drift)
# --------------------------------------------------------------------------------------


def _check_backbone(backbone) -> None:
    if int(backbone.cfg.num_register_tokens) != 0:
        raise NotImplementedError("the fused layout assumes a register-free DINOv2 backbone")


def build_patch_weight(backbone) -> torch.Tensor:
    """Flattened patch-embed conv kernel (ps*ps*3 -> embed_dim), zero-extended to K_pad rows.

    Row order ``(kh, kw, c)`` matches :func:`im2col_patches`; rows >= ps*ps*3 are zero so the
    zero pad columns of the im2col contribute exact zeros.
    """
    w = backbone.patch_embed_proj.weight.detach()  # (D, 3, ps, ps)
    d = w.shape[0]
    k = w.shape[1] * w.shape[2] * w.shape[3]
    w_flat = w.permute(2, 3, 1, 0).reshape(k, d)
    out = torch.zeros(pad_to_tile(k), d, dtype=torch.float32)
    out[:k] = w_flat.float()
    return out


def build_patch_const(backbone) -> torch.Tensor:
    """C_patch (1, S, D): rows 0..num_patches-1 = pos_patch + conv_bias, row num_patches = cls + pos_cls."""
    _check_backbone(backbone)
    pos = backbone.pos_embed.detach().float()  # (1, 1 + P, D)
    bias = backbone.patch_embed_proj.bias.detach().float().reshape(1, 1, -1)
    cls = backbone.cls_token.detach().float()  # (1, 1, D)
    return torch.cat([pos[:, 1:] + bias, cls + pos[:, :1]], dim=1).contiguous()


def build_gated_proj_consts(ref) -> Tuple[torch.Tensor, torch.Tensor]:
    """(C2, G), both (1, S, dim) fp32 for ``X_base = C2 + (x_final @ W_proj) * G``.

    C2 rows 0..P-1 = gaze pos-embed + proj bias, row P = inout token; G = 1 on rows 0..P-1, 0 on row P.
    """
    if not getattr(ref, "inout", False):
        raise NotImplementedError("the fused layout needs the in/out token (inout=True)")
    dim = ref.dim
    pe = ref.pos_embed.detach().float().permute(1, 2, 0).reshape(1, -1, dim)  # (1, P, dim)
    bias = ref.linear.bias.detach().float().reshape(1, 1, dim)
    inout_tok = ref.inout_token.weight.detach().float().reshape(1, 1, dim)
    c2 = torch.cat([pe + bias, inout_tok], dim=1).contiguous()
    g = torch.ones(1, c2.shape[1], dim, dtype=torch.float32)
    g[:, -1] = 0.0
    return c2, g


def build_proj_weight(ref) -> torch.Tensor:
    """(embed_dim, dim) weight of the 1x1 projection conv as a matmul weight."""
    return ref.linear.weight.detach().float().squeeze(-1).squeeze(-1).T.contiguous()


def build_heatmap_head(ref) -> Tuple[torch.Tensor, float]:
    """Fused ConvTranspose2d(dim,dim,2,2) + Conv2d(dim,1,1,bias=False) -> ((dim, 4) weight, scalar bias).

    Column ``2*a + b`` is the (a, b) sub-pixel of the 2x2 block a patch expands to (the legacy formula).
    """
    ct_w = ref.heatmap_head[0].weight.detach().float()  # (in=dim, out=dim, 2, 2)
    ct_b = ref.heatmap_head[0].bias.detach().float()
    c1_w = ref.heatmap_head[1].weight.detach().float().squeeze(-1).squeeze(-1)  # (1, dim)
    w_fused = torch.einsum("ko,ioab->ikab", c1_w, ct_w).squeeze(1)  # (dim, 2, 2)
    b_fused = float((c1_w @ ct_b).squeeze().item())
    return w_fused.reshape(ref.dim, 4).contiguous(), b_fused


def build_inout_head(ref) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(W1 (dim,128), b1 (1,128), W2 (128,1), b2 (1,1)) of ``inout_head`` = Linear, ReLU, Dropout, Linear, Sigmoid."""
    if not getattr(ref, "inout", False):
        raise NotImplementedError("the fused layout needs the in/out head (inout=True)")
    w1 = ref.inout_head[0].weight.detach().float().T.contiguous()
    b1 = ref.inout_head[0].bias.detach().float().reshape(1, -1)
    w2 = ref.inout_head[3].weight.detach().float().T.contiguous()
    b2 = ref.inout_head[3].bias.detach().float().reshape(1, -1)
    return w1, b1, w2, b2


def build_fused_head(ref) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-diagonal head weight (dim + 128, 5) and bias (1, 5) for ``sigmoid([X, h] @ W + b)``.

    Rows 0..dim-1 / cols 0..3 = the fused heatmap weight, rows dim..dim+127 / col 4 = the second
    in/out linear; the off-diagonal blocks are zero (exact zeros in the K-sum). The K-concat seam
    at ``dim`` is tile-aligned (256 and 128 are multiples of 32), which is what the device op needs.
    """
    w_hm, b_hm = build_heatmap_head(ref)
    _, _, w2, b2 = build_inout_head(ref)
    dim, hid = w_hm.shape[0], w2.shape[0]
    if dim % TILE or hid % TILE:
        raise ValueError("fused head needs tile-aligned K segments")
    w = torch.zeros(dim + hid, 5, dtype=torch.float32)
    w[:dim, :4] = w_hm
    w[dim:, 4] = w2[:, 0]
    b = torch.zeros(1, 5, dtype=torch.float32)
    b[0, :4] = b_hm
    b[0, 4] = b2[0, 0]
    return w, b


def legacy_to_fused_order(x: torch.Tensor) -> torch.Tensor:
    """(B, S, C) with the special token at row 0 -> at row S-1 (the fused sequence order)."""
    return torch.cat([x[:, 1:], x[:, :1]], dim=1)


def fused_to_legacy_order(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`legacy_to_fused_order`."""
    return torch.cat([x[:, -1:], x[:, :-1]], dim=1)


# --------------------------------------------------------------------------------------
# fp32 torch mirror of the fused device graph (tests + device-pass shadow)
# --------------------------------------------------------------------------------------


@torch.no_grad()
def torch_fused_forward(ref, images: torch.Tensor, bboxes: Sequence[Sequence[float]],
                        capture: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    """Run the fused layout in fp32 torch with the SAME constant tables the device path uploads.

    Returns ``{"heatmap": (N, 64, 64), "inout": (N,)}`` like the reference; ``capture`` (optional
    dict) receives the fused-order intermediates ``after_prefix``, ``after_block_i``,
    ``after_final_norm``, ``x_base``, ``x0``, ``after_gaze_blocks``, ``head_out``.
    """
    backbone = ref.backbone
    _check_backbone(backbone)
    n_side = backbone.img_size // backbone.patch_size
    ps = backbone.patch_size
    k = ps * ps * 3
    num_patches = n_side * n_side
    seq = num_patches + 1

    def cap(key, t):
        if capture is not None:
            capture[key] = t.detach().clone()

    x_pad = torch.zeros(1, seq, pad_to_tile(k), dtype=torch.float32)
    fill_im2col(x_pad, images.float(), n_side, ps)
    x = x_pad @ build_patch_weight(backbone) + build_patch_const(backbone)
    cap("after_prefix", x)
    for i, blk in enumerate(backbone.blocks):
        x = blk(x)
        cap(f"after_block_{i}", x)
    x = backbone.norm(x)
    cap("after_final_norm", x)

    c2, g = build_gated_proj_consts(ref)
    x_base = c2 + (x @ build_proj_weight(ref)) * g  # (1, S, dim)
    cap("x_base", x_base)

    mask = build_head_mask(bboxes, ref.featmap_h, ref.featmap_w, seq)  # (N, S, 1)
    x0 = x_base + mask * ref.head_token.weight.detach().float().reshape(1, 1, -1)
    cap("x0", x0)
    for gb in ref.transformer:
        x0 = gb(x0)
    cap("after_gaze_blocks", x0)

    w1, b1, _, _ = build_inout_head(ref)
    h = F.relu(x0 @ w1 + b1)
    w_head, b_head = build_fused_head(ref)
    out = torch.sigmoid(torch.cat([x0, h], dim=-1) @ w_head + b_head)  # (N, S, 5)
    cap("head_out", out)
    heatmap, inout = decode_head_output(out, ref.featmap_h, ref.featmap_w, len(bboxes), ref.out_size)
    return {"heatmap": heatmap, "inout": inout}
