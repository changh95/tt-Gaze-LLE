# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TT-NN Gaze-LLE, end-to-end on a single Blackhole p150a chip.

Everything between the first matmul and the final sigmoid runs on device.
Host-side work is pure layout, not inference compute:
  * `(B, 3, H, W) -> (B, num_patches, ps*ps*3)` view+permute+reshape of the image,
  * building the tiny (num_patches,) binary head bbox mask,
  * upload / download,
  * a `(B, num_patches, 4) -> (B, 64, 64)` view on the downloaded heatmap.

Device placement:
  * Patch embedding via a direct `(ps*ps*3, embed_dim)` matmul.
  * Pre-composed [CLS+pos_cls, REG] prefix + patch pos_embed add + concat.
  * DINOv2 ViT-B/14 encoder (12 blocks) + final LayerNorm + slice (CLS+REG).
  * 1x1 projection 768->256, gaze pos + head_map*head_token conditioning.
  * 3 gaze decoder blocks.
  * Fused ConvTranspose2d(256,256,2,2) + Conv2d(256,1,1) + Sigmoid heatmap head,
    expressed as a single (256,4) matmul + scalar add + sigmoid.
  * In/out MLP (256->128->1) with ReLU + Sigmoid.

The above is the LEGACY path (``TT_FUSED=0``; eager, per-head decoder, 201 device ops at N=1 and
57 more per extra head, two readbacks per head). It is bit-for-bit what shipped in the first
package and is not touched by the knob below.

The fused pipeline below is the DEFAULT since its device validation on 2026-09-13
(``DEVICE_VALIDATION.md`` "Results"; ``TT_FUSED`` unset or ``1``; read ONCE in
``TtGazeLLE.__init__``, see ``fused_math.FusedConfig``), built for the whole-graph metal trace:

  * Host: bf16 im2col into a persistent ROW_MAJOR (1, 1025, 608) buffer (zero CLS row 1024,
    zero pad cols) -> ``copy_host_to_device_tensor``; device ``tilize_with_zero_padding``
    replaces the ~0.8 ms host tilize (exact).
  * Token layout ``[patches, CLS]`` / ``[patches, inout]`` (special token at row 1024, exact by
    permutation equivariance) so the tile-padded ``concat`` (4 ops) and unaligned ``slice``
    (3 ops) fallbacks disappear.
  * Patch-embed + conv bias + pos-embed + CLS as ONE ``dit_minimal_matmul_addcmul_fused``
    (``C_patch + 1.0 * (X @ W_patch) * ones``); proj+residual and fc2+residual of every backbone
    and gaze block as one fused op each (11 -> 9 ops per block); the 768->256 projection +
    gaze pos + bias + inout token as ONE gated fused op (``C2 + (x @ W_proj) * G``).
  * Head conditioning from a host-built (N, 1025, 1) mask (``mul`` + ``add``), and all N heads
    batched through the decoder once as (N, 1025, 256): 57*N -> 32 ops for any N.
  * Heads as 3 ops: ``relu(x @ W1 + b1)``, ``minimal_matmul([x, h], blockdiag(W_hm, W_io2), bias)``,
    ``sigmoid`` -> ONE (N, 1025, 5) readback (heatmap rows 0..1023 cols 0..3, in/out row 1024
    col 4). ``TT_FUSED_HEAD=fused`` folds the sigmoid into the matmul (measured no faster).
  * ``SDPAProgramConfig`` chunks (q 128 / k 128, exact exp) instead of the 32/32 defaults.
  * Scene trace (once) + one decoder trace per head-count bucket (``TT_FUSED_TRACE_HEADS``,
    default 1,2,3,4,6,8,10; other N are padded up by repeating the last bbox); ``warmup()``
    captures them before the server reports READY. 144 device ops for any N (legacy 201/315/714
    at N = 1/3/10), 1 upload + 1 mask upload + 1 readback. ``TT_FUSED_EAGER=1`` runs the same
    graph eagerly; every fusion has a legacy-op fallback knob (``fused_math.FusedConfig``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

import ttnn

from gaze_lle.tt.fused_math import (
    FusedConfig,
    build_fused_head,
    build_gated_proj_consts,
    build_head_mask,
    build_heatmap_head,
    build_inout_head,
    build_patch_const,
    build_patch_weight,
    build_proj_weight,
    decode_head_output,
    decode_legacy_head_outputs,
    fill_im2col,
    fused_to_legacy_order,
    pad_bboxes,
    pad_to_tile,
    pick_bucket,
    select_forward_path,
)


def open_device_kwargs(cfg: Optional[FusedConfig] = None) -> dict:
    """Extra ``ttnn.open_device`` kwargs for this model.

    Legacy path (``TT_FUSED=0``): ``{}`` -- its validated recipe is a bare ``open_device(device_id=...)``.
    Fused path (default) with tracing on: ``{"trace_region_size": TT_FUSED_TRACE_REGION_MB << 20}``,
    because ``ttnn.device.DEFAULT_TRACE_REGION_SIZE`` is 0 in this tree and a bare device
    cannot capture a trace. Every device-opening entry point (server, benchmark, conftest,
    scripts) spreads this into its ``open_device`` call.
    """
    cfg = FusedConfig.from_env() if cfg is None else cfg
    if cfg.enabled and cfg.trace:
        return {"trace_region_size": int(cfg.trace_region_mb) * 1024 * 1024}
    return {}


def _to_device(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


# p150a Blackhole has a 10x13 compute grid; openvla uses 10x12 and it is validated.
_CORE_GRID = ttnn.CoreGrid(y=10, x=13)

# LoFi matmul for bfp8_b weights: 2x throughput vs HiFi4, small PCC impact.
_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)


class _BlockParams:
    """Holds on-device weights for one DINOv2 block.

    LayerScale is folded into the preceding projection (``ls1 -> attn.proj``,
    ``ls2 -> mlp.fc2``) so each block drops two elementwise multiplies.
    """

    def __init__(self, block, device):
        # Attention
        self.norm1_w = _to_device(block.norm1.weight.unsqueeze(0), device)
        self.norm1_b = _to_device(block.norm1.bias.unsqueeze(0), device)
        self.qkv_w = _to_device(block.attn.qkv.weight.T.contiguous(), device, dtype=ttnn.bfloat8_b)
        self.qkv_b = _to_device(block.attn.qkv.bias.unsqueeze(0), device)
        ls1 = block.ls1.gamma.detach()
        proj_w = block.attn.proj.weight.detach() * ls1.unsqueeze(-1)
        proj_b = block.attn.proj.bias.detach() * ls1
        self.proj_w = _to_device(proj_w.T.contiguous(), device, dtype=ttnn.bfloat8_b)
        self.proj_b = _to_device(proj_b.unsqueeze(0), device)

        # MLP
        self.norm2_w = _to_device(block.norm2.weight.unsqueeze(0), device)
        self.norm2_b = _to_device(block.norm2.bias.unsqueeze(0), device)
        self.fc1_w = _to_device(block.mlp.fc1.weight.T.contiguous(), device, dtype=ttnn.bfloat8_b)
        self.fc1_b = _to_device(block.mlp.fc1.bias.unsqueeze(0), device)
        ls2 = block.ls2.gamma.detach()
        fc2_w = block.mlp.fc2.weight.detach() * ls2.unsqueeze(-1)
        fc2_b = block.mlp.fc2.bias.detach() * ls2
        self.fc2_w = _to_device(fc2_w.T.contiguous(), device, dtype=ttnn.bfloat8_b)
        self.fc2_b = _to_device(fc2_b.unsqueeze(0), device)


def _dinov2_attention(x, p: _BlockParams, num_heads: int):
    hidden_states = ttnn.layer_norm(x, weight=p.norm1_w, bias=p.norm1_b, epsilon=1e-6)
    head_dim = hidden_states.shape[-1] // num_heads
    qkv = ttnn.linear(
        hidden_states, p.qkv_w, bias=p.qkv_b, core_grid=_CORE_GRID, compute_kernel_config=_LOFI
    )
    ttnn.deallocate(hidden_states)
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=num_heads, transpose_key=False
    )
    ttnn.deallocate(qkv)
    ctx = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=1.0 / (head_dim**0.5)
    )
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    ctx = ttnn.transformer.concatenate_heads(ctx)

    out = ttnn.linear(ctx, p.proj_w, bias=p.proj_b, core_grid=_CORE_GRID, compute_kernel_config=_LOFI)
    ttnn.deallocate(ctx)
    return ttnn.add(x, out)


def _dinov2_mlp(x, p: _BlockParams):
    hidden_states = ttnn.layer_norm(x, weight=p.norm2_w, bias=p.norm2_b, epsilon=1e-6)
    h = ttnn.linear(
        hidden_states, p.fc1_w, bias=p.fc1_b, activation="gelu",
        core_grid=_CORE_GRID, compute_kernel_config=_LOFI,
    )
    ttnn.deallocate(hidden_states)
    h2 = ttnn.linear(h, p.fc2_w, bias=p.fc2_b, core_grid=_CORE_GRID, compute_kernel_config=_LOFI)
    ttnn.deallocate(h)
    return ttnn.add(x, h2)


def _dinov2_block(x, p: _BlockParams, num_heads: int):
    x = _dinov2_attention(x, p, num_heads)
    x = _dinov2_mlp(x, p)
    return x


class _GazeBlockParams:
    """Holds on-device weights for one gaze-decoder transformer block (no LayerScale).

    The official Gaze-LLE checkpoint uses ``qkv_bias=False`` for the gaze decoder's
    attention, so we tolerate ``block.attn.qkv.bias`` being ``None``.
    """

    def __init__(self, block, device):
        self.norm1_w = _to_device(block.norm1.weight.unsqueeze(0), device)
        self.norm1_b = _to_device(block.norm1.bias.unsqueeze(0), device)
        self.qkv_w = _to_device(block.attn.qkv.weight.T.contiguous(), device)
        self.qkv_b = (
            _to_device(block.attn.qkv.bias.unsqueeze(0), device)
            if block.attn.qkv.bias is not None
            else None
        )
        self.proj_w = _to_device(block.attn.proj.weight.T.contiguous(), device)
        self.proj_b = _to_device(block.attn.proj.bias.unsqueeze(0), device)
        self.norm2_w = _to_device(block.norm2.weight.unsqueeze(0), device)
        self.norm2_b = _to_device(block.norm2.bias.unsqueeze(0), device)
        self.fc1_w = _to_device(block.mlp.fc1.weight.T.contiguous(), device)
        self.fc1_b = _to_device(block.mlp.fc1.bias.unsqueeze(0), device)
        self.fc2_w = _to_device(block.mlp.fc2.weight.T.contiguous(), device)
        self.fc2_b = _to_device(block.mlp.fc2.bias.unsqueeze(0), device)


def _gaze_block(x, p: _GazeBlockParams, num_heads: int):
    # Attention via fused scaled-dot-product-attention kernel.
    hidden_states = ttnn.layer_norm(x, weight=p.norm1_w, bias=p.norm1_b, epsilon=1e-6)
    qkv = ttnn.linear(hidden_states, p.qkv_w, bias=p.qkv_b, core_grid=_CORE_GRID)
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=num_heads, transpose_key=False
    )
    ttnn.deallocate(qkv)
    head_dim = hidden_states.shape[-1] // num_heads
    ctx = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=1.0 / (head_dim**0.5)
    )
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    ctx = ttnn.transformer.concatenate_heads(ctx)
    out = ttnn.linear(ctx, p.proj_w, bias=p.proj_b, core_grid=_CORE_GRID)
    ttnn.deallocate(ctx)
    x = ttnn.add(x, out)

    # MLP
    hidden_states = ttnn.layer_norm(x, weight=p.norm2_w, bias=p.norm2_b, epsilon=1e-6)
    h = ttnn.linear(hidden_states, p.fc1_w, bias=p.fc1_b, activation="gelu", core_grid=_CORE_GRID)
    h = ttnn.linear(h, p.fc2_w, bias=p.fc2_b, core_grid=_CORE_GRID)
    return ttnn.add(x, h)


# ======================================================================================
# TT_FUSED=1 path. The legacy classes/functions above and TtGazeLLE below are unchanged
# except for the knob dispatch in TtGazeLLE.__init__ / __call__ (and the mixin base class).
# ======================================================================================


def _compute_config(device, fidelity: str, fp32_acc: bool = False, packer_l1_acc: bool = False):
    """Explicit compute config (the fused ops default to HiFi2 + fp32 accumulation otherwise)."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


class _FusedRuntime:
    """Device-side knobs of the fused graph: memory placement, kernel configs, fused-op helpers.

    Constraints baked in (from the op sources of this tree, see DEVICE_VALIDATION.md):
      * ``dit_minimal_matmul_addcmul_fused``: residual (ternary_a) data format == weight format
        (program factory ``ternary_a_data_format == in1_data_format``) -> bf16 weights for every
        fused matmul; residual and scale vector in the SAME buffer type (L1 / DRAM); explicit
        4x4x4 blocks / 2x2 subblocks (the 8x8x8 default clashed L1 on p150 for rf-detr).
      * ``minimal_matmul`` fused K-concat: exactly [prefix, suffix], same dtype/leading dims,
        prefix_padded_K + suffix_padded_K == weight_padded_K; fused_activation and ternary are
        mutually exclusive.
      * SDPA: chunk sizes % 32 == 0; no explicit padding mask (the kernel masks keys >= logical S).
    """

    def __init__(self, device, cfg: FusedConfig):
        self.device = device
        self.cfg = cfg
        self.mem = ttnn.L1_MEMORY_CONFIG if cfg.l1 else None
        grid = device.compute_with_storage_grid_size()
        self.grid = grid
        # Fused matmuls: LoFi like the legacy linears unless TT_FUSED_FIDELITY says otherwise.
        self.mm_compute = _LOFI if cfg.fidelity == "LoFi" else _compute_config(device, cfg.fidelity)
        # Heads: HiFi2 (the legacy in/out linears ran HiFi2: no user grid, bf16 inputs).
        self.head_compute = _compute_config(device, "HiFi2")
        self.dit_config = ttnn.MinimalMatmulConfig(  # proj [.,768]x[768,768], fc2 [.,3072]x[3072,768], patch, gated proj
            M_block_size=4, K_block_size=4, N_block_size=4, subblock_h=2, subblock_w=2,
            compute_with_storage_grid_size=grid,
        )
        self.mm_config = ttnn.MinimalMatmulConfig(  # qkv [.,768]x[768,2304], fc1 [.,768]x[768,3072] (TT_FUSED_MINIMAL_MM=1)
            M_block_size=8, K_block_size=4, N_block_size=4, subblock_h=2, subblock_w=2,
            compute_with_storage_grid_size=grid,
        )
        self.sdpa_pc = (
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid,
                q_chunk_size=cfg.sdpa_q,
                k_chunk_size=cfg.sdpa_k,
                exp_approx_mode=not cfg.sdpa_exact_exp,
            )
            if cfg.sdpa_pc
            else None
        )
        # Fused sigmoid params = ttnn.sigmoid defaults (vector mode RC=4, accurate mode): the same
        # SFPU kernel, so "fused" and "separate" should agree; A/B on device (TT_FUSED_HEAD).
        self.sigmoid_act = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID, 4.0, 0.0)
        self._ones_dram: Dict[int, ttnn.Tensor] = {}
        self._ones_l1: Dict[int, ttnn.Tensor] = {}
        self._frozen = False

    # ---- constants -------------------------------------------------------------------
    def preallocate(self, widths: Sequence[int]) -> None:
        """Materialise every constant BEFORE any trace capture, then refuse new ones.

        A device buffer allocated AFTER a capture can land on an address that trace's (freed)
        intermediates use, and the next replay of that trace overwrites it. Measured on the p150a
        (2026-09-13): ``ones(256)`` created lazily inside the decoder warm-up run, i.e. after the
        scene capture, was clobbered by the scene replay -> gaze-block PCC 0.76 / heatmap 0.57 on
        every later call (eager or traced), while ``ones(768)`` (created before the capture) was fine.
        """
        for w in widths:
            self.ones(int(w))
            if self.cfg.l1:
                self.ones(int(w), l1=True)
        self._frozen = True

    def ones(self, width: int, l1: bool = False) -> ttnn.Tensor:
        """(1, 1, width) bf16 ones vector = the addcmul scale of a plain residual add."""
        table = self._ones_l1 if l1 else self._ones_dram
        if width not in table:
            if self._frozen:
                raise RuntimeError(
                    f"ones({width}, l1={l1}) requested after preallocate(): allocating a constant after a "
                    "trace capture can alias the trace's intermediates (see preallocate); add the width there"
                )
            if l1:
                table[width] = ttnn.to_memory_config(self.ones(width, l1=False), ttnn.L1_MEMORY_CONFIG)
            else:
                table[width] = ttnn.from_torch(
                    torch.ones(1, 1, width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
        return table[width]

    def ones_for(self, residual: ttnn.Tensor, width: int) -> ttnn.Tensor:
        """Ones vector in the residual's buffer type (the fused kernel compiles ONE accessor type)."""
        in_l1 = residual.memory_config().buffer_type == ttnn.BufferType.L1
        return self.ones(width, l1=in_l1)

    # ---- fused-op helpers ------------------------------------------------------------
    def matmul(self, h, w, b, activation=None, compute=None):
        """h @ w + b [+ activation] (qkv, fc1): legacy ``ttnn.linear`` or ``minimal_matmul`` (knob)."""
        if self.cfg.minimal_mm:
            y = ttnn.experimental.minimal_matmul(
                h, w, bias_tensor=b, config=self.mm_config, memory_config=self.mem,
                compute_kernel_config=self.mm_compute,
            )
            if activation == "gelu":
                y = ttnn.gelu(y, fast_and_approximate_mode=False, memory_config=self.mem)  # exact, like linear's "gelu"
            elif activation is not None:
                raise ValueError(f"unsupported activation {activation!r} for the minimal_matmul path")
            return y
        return ttnn.linear(
            h, w, bias=b, activation=activation, core_grid=_CORE_GRID,
            compute_kernel_config=compute, memory_config=self.mem,
        )

    def matmul_residual(self, h, w, b, residual, compute=None):
        """residual + (h @ w + b) (proj, fc2): one fused op, or linear + add (TT_FUSED_DIT=0)."""
        if self.cfg.dit:
            return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                h, w, 1.0, residual, self.ones_for(residual, w.shape[-1]),
                bias_tensor=b, config=self.dit_config, memory_config=self.mem,
                compute_kernel_config=self.mm_compute,
            )
        y = ttnn.linear(h, w, bias=b, core_grid=_CORE_GRID, compute_kernel_config=compute, memory_config=self.mem)
        return ttnn.add(residual, y, memory_config=self.mem)


class _FusedBlockParams:
    """On-device weights of one DINOv2 block for the fused path (LayerScale folded like legacy).

    qkv / fc1 stay bfp8 (bf16 with TT_FUSED_BF16_WEIGHTS=1); proj / fc2 are bf16 whenever the
    fused matmul+residual op is on (its residual/weight data formats must match), else bfp8 like legacy.
    """

    def __init__(self, block, device, cfg: FusedConfig):
        wdt = ttnn.bfloat16 if cfg.bf16_weights else ttnn.bfloat8_b
        res_wdt = ttnn.bfloat16 if cfg.dit else wdt
        self.norm1_w = _to_device(block.norm1.weight.unsqueeze(0), device)
        self.norm1_b = _to_device(block.norm1.bias.unsqueeze(0), device)
        self.qkv_w = _to_device(block.attn.qkv.weight.T.contiguous(), device, dtype=wdt)
        self.qkv_b = _to_device(block.attn.qkv.bias.unsqueeze(0), device)
        ls1 = block.ls1.gamma.detach()
        proj_w = block.attn.proj.weight.detach() * ls1.unsqueeze(-1)
        proj_b = block.attn.proj.bias.detach() * ls1
        self.proj_w = _to_device(proj_w.T.contiguous(), device, dtype=res_wdt)
        self.proj_b = _to_device(proj_b.unsqueeze(0), device)
        self.norm2_w = _to_device(block.norm2.weight.unsqueeze(0), device)
        self.norm2_b = _to_device(block.norm2.bias.unsqueeze(0), device)
        self.fc1_w = _to_device(block.mlp.fc1.weight.T.contiguous(), device, dtype=wdt)
        self.fc1_b = _to_device(block.mlp.fc1.bias.unsqueeze(0), device)
        ls2 = block.ls2.gamma.detach()
        fc2_w = block.mlp.fc2.weight.detach() * ls2.unsqueeze(-1)
        fc2_b = block.mlp.fc2.bias.detach() * ls2
        self.fc2_w = _to_device(fc2_w.T.contiguous(), device, dtype=res_wdt)
        self.fc2_b = _to_device(fc2_b.unsqueeze(0), device)


def _fused_dinov2_block(x, p: _FusedBlockParams, rt: _FusedRuntime, num_heads: int):
    """One DINOv2 block on (1, 1025, 768) in the fused layout: 9 ops (11 legacy).

    LN, qkv, split heads, SDPA (program config), concat heads, proj+residual (fused),
    LN, fc1+gelu, fc2+residual (fused).
    """
    mem = rt.mem
    h = ttnn.layer_norm(x, weight=p.norm1_w, bias=p.norm1_b, epsilon=1e-6, memory_config=mem)
    head_dim = h.shape[-1] // num_heads
    qkv = rt.matmul(h, p.qkv_w, p.qkv_b, compute=_LOFI)
    ttnn.deallocate(h)
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=num_heads, transpose_key=False, memory_config=mem
    )
    ttnn.deallocate(qkv)
    ctx = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=1.0 / (head_dim ** 0.5), program_config=rt.sdpa_pc, memory_config=mem
    )
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    ctx = ttnn.transformer.concatenate_heads(ctx, memory_config=mem)
    x = rt.matmul_residual(ctx, p.proj_w, p.proj_b, x, compute=_LOFI)
    ttnn.deallocate(ctx)

    h = ttnn.layer_norm(x, weight=p.norm2_w, bias=p.norm2_b, epsilon=1e-6, memory_config=mem)
    h = rt.matmul(h, p.fc1_w, p.fc1_b, activation="gelu", compute=_LOFI)
    x = rt.matmul_residual(h, p.fc2_w, p.fc2_b, x, compute=_LOFI)
    ttnn.deallocate(h)
    return x


class _FusedGazeBlockParams(_GazeBlockParams):
    """Gaze-decoder block weights for the fused path (all bf16 already, same as legacy)."""

    def __init__(self, block, device, cfg: FusedConfig):
        super().__init__(block, device)


def _fused_gaze_block(x, p: _GazeBlockParams, rt: _FusedRuntime, num_heads: int):
    """One gaze block on (N, 1025, 256), all N heads batched: 9 ops (11 legacy).

    The non-fused matmuls are called exactly like the legacy block (core_grid, no explicit
    compute config => LoFi) so an A/B isolates the fused ops.
    """
    mem = rt.mem
    h = ttnn.layer_norm(x, weight=p.norm1_w, bias=p.norm1_b, epsilon=1e-6, memory_config=mem)
    head_dim = h.shape[-1] // num_heads
    qkv = rt.matmul(h, p.qkv_w, p.qkv_b)
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=num_heads, transpose_key=False, memory_config=mem
    )
    ttnn.deallocate(qkv)
    ctx = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=1.0 / (head_dim ** 0.5), program_config=rt.sdpa_pc, memory_config=mem
    )
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    ctx = ttnn.transformer.concatenate_heads(ctx, memory_config=mem)
    x = rt.matmul_residual(ctx, p.proj_w, p.proj_b, x)
    ttnn.deallocate(ctx)

    h = ttnn.layer_norm(x, weight=p.norm2_w, bias=p.norm2_b, epsilon=1e-6, memory_config=mem)
    h = rt.matmul(h, p.fc1_w, p.fc1_b, activation="gelu")
    x = rt.matmul_residual(h, p.fc2_w, p.fc2_b, x)
    ttnn.deallocate(h)
    return x


WARMUP_BBOX = (0.3, 0.2, 0.6, 0.5)  # the bbox every harness in the port warms with


class _FusedMixin:
    """The TT_FUSED=1 implementation of :class:`TtGazeLLE` (its base class; see __init__ dispatch).

    Persistent device state and the trace contract:

      * ``_in_dev``      ROW_MAJOR bf16 (1, 1025, 608) im2col upload buffer (allocated first).
      * ``_x_base``      the scene trace's output (1, 1025, 256) -- lives as long as the trace.
      * ``_dec[n]``      per head-count bucket: the decoder trace id, its TILE bf16 (n, 1025, 1)
                         mask upload buffer and its output tensor(s).

    Order per inference (must not change): upload im2col -> execute scene trace -> upload mask ->
    execute decoder trace -> read outputs. Intermediates of a captured trace live at baked
    addresses that later-allocated persistent buffers (other buckets' masks/outputs) may reuse, so
    every persistent buffer is (re)written after the trace that could clobber it and read before
    the next one runs. Nothing is allocated between capture and replay except through this class.
    """

    # ------------------------------------------------------------------ construction
    def _init_fused(self, ref_model, device):
        cfg: FusedConfig = self.fused_cfg
        self._rt = _FusedRuntime(device, cfg)
        backbone = ref_model.backbone
        if hasattr(device, "enable_program_cache"):
            device.enable_program_cache()

        self.seq_len = self.num_patches + 1  # patches + one special token (row num_patches)
        self.k_patch = self.patch_size * self.patch_size * 3
        self.k_patch_pad = pad_to_tile(self.k_patch)

        self.block_params = [_FusedBlockParams(blk, device, cfg) for blk in backbone.blocks]
        self.final_norm_w = _to_device(backbone.norm.weight.unsqueeze(0), device)
        self.final_norm_b = _to_device(backbone.norm.bias.unsqueeze(0), device)

        # Patch embed as ONE fused op: C_patch + 1.0 * (X_pad @ W_pad) * ones. W must be bf16 when
        # fused (format rule); the linear+add fallback keeps the legacy bfp8 weight.
        self.patch_embed_w = _to_device(build_patch_weight(backbone), device,
                                        dtype=ttnn.bfloat16 if cfg.dit else ttnn.bfloat8_b)
        self.patch_const = _to_device(build_patch_const(backbone), device)  # (1, S, 768), DRAM
        self.patch_embed_b = None  # folded into patch_const
        self.prefix_tt = None      # no concat in the fused layout
        self.pos_patches_tt = None

        # Gated projection: X_base = C2 + 1.0 * (x_final @ W_proj) * G.
        self.proj_w = _to_device(build_proj_weight(ref_model), device)
        c2, gate = build_gated_proj_consts(ref_model)
        self.proj_c2 = _to_device(c2, device)      # (1, S, 256), DRAM (ternary_a)
        self.proj_gate = _to_device(gate, device)  # (1, S, 256), DRAM (ternary_b, full [M, N])
        self.head_token_tt = _to_device(ref_model.head_token.weight.unsqueeze(0), device)  # (1, 1, 256)

        self.gaze_block_params = [_FusedGazeBlockParams(blk, device, cfg) for blk in ref_model.transformer]

        w1, b1, w2, b2 = build_inout_head(ref_model)
        self.inout_fc1_w = _to_device(w1, device)
        self.inout_fc1_b = _to_device(b1, device)
        if cfg.head == "legacy":
            self.inout_fc2_w = _to_device(w2, device)
            self.inout_fc2_b = _to_device(b2, device)
            hm_w, hm_b = build_heatmap_head(ref_model)
            self.heatmap_w = _to_device(hm_w, device)
            self.heatmap_b_tt = _to_device(torch.full((1, 1, 1), float(hm_b)), device)
        else:
            w_head, b_head = build_fused_head(ref_model)
            self.head_w = _to_device(w_head, device)  # (384, 5) block-diagonal, bf16
            self.head_b = _to_device(b_head, device)  # (1, 5)

        # Every device constant must exist BEFORE the first trace capture (allocator aliasing, see
        # _FusedRuntime.preallocate): the ones vectors of the backbone (768) and decoder (256) widths.
        self._rt.preallocate((self.embed_dim, self.dim))

        # Persistent host im2col buffer: bf16, zero CLS row and zero pad columns never written.
        self._im2col_buf = torch.zeros(1, self.seq_len, self.k_patch_pad, dtype=torch.bfloat16)
        self._in_dev = None
        self._x_base = None
        self._scene_trace = None
        self._dec: Dict[int, dict] = {}

    # ------------------------------------------------------------------ host work
    def _host_input(self, images: torch.Tensor) -> ttnn.Tensor:
        """im2col into the persistent bf16 buffer -> host ROW_MAJOR ttnn tensor (no host tilize)."""
        fill_im2col(self._im2col_buf, images.float(), self.num_patches_side, self.patch_size)
        return ttnn.from_torch(self._im2col_buf, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _host_mask(self, bboxes) -> ttnn.Tensor:
        """(N, S, 1) bf16 TILE head mask (row S-1 = 0), tilized on host (68 KB per head)."""
        m = build_head_mask(bboxes, self.featmap_h, self.featmap_w, self.seq_len).to(torch.bfloat16)
        return ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # ------------------------------------------------------------------ device graph
    def _fused_scene_graph(self, in_dev, cap=None):
        """Persistent RM im2col buffer -> X_base (1, S, 256): tilize + fused patch-embed + 12 blocks
        + final LN + gated projection. 1 + 1 + 12*9 + 1 + 1 = 112 ops (legacy 144 + host tilize)."""
        rt = self._rt
        mem = rt.mem
        x = ttnn.tilize_with_zero_padding(in_dev, memory_config=mem, use_multicore=True)  # (1, S, 608) TILE
        if rt.cfg.dit:
            x = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                x, self.patch_embed_w, 1.0, self.patch_const, rt.ones(self.embed_dim),
                bias_tensor=None, config=rt.dit_config, memory_config=mem, compute_kernel_config=rt.mm_compute,
            )
        else:
            y = ttnn.linear(x, self.patch_embed_w, bias=None, core_grid=_CORE_GRID,
                            compute_kernel_config=_LOFI, memory_config=mem)
            x = ttnn.add(y, self.patch_const, memory_config=mem)
        if cap:
            cap("after_prefix", x, order="fused")

        for i, bp in enumerate(self.block_params):
            x = _fused_dinov2_block(x, bp, rt, self.num_heads)
            if cap and i in (0, 5, 11):
                cap(f"after_block_{i}", x, order="fused")
        x = ttnn.layer_norm(x, weight=self.final_norm_w, bias=self.final_norm_b, epsilon=1e-6, memory_config=mem)
        if cap:
            cap("after_final_norm", x, order="fused")

        if rt.cfg.dit:
            x_base = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                x, self.proj_w, 1.0, self.proj_c2, self.proj_gate,
                bias_tensor=None, config=rt.dit_config, memory_config=mem, compute_kernel_config=rt.mm_compute,
            )
        else:
            y = ttnn.linear(x, self.proj_w, bias=None, core_grid=_CORE_GRID, memory_config=mem)
            y = ttnn.mul(y, self.proj_gate, memory_config=mem)
            x_base = ttnn.add(y, self.proj_c2, memory_config=mem)
        if cap:
            cap("after_gaze_proj_pos", x_base, rows=self.num_patches)
        return x_base

    def _fused_decoder_graph(self, mask_dev, x_base, cap=None):
        """(N, S, 1) mask + X_base -> head output(s): mul, add, 3 x 9-op gaze block, relu-linear,
        fused head matmul (+ optional separate sigmoid) = 31-32 ops for any N (legacy 57 per head)."""
        rt = self._rt
        cfg = rt.cfg
        mem = rt.mem
        contrib = ttnn.mul(mask_dev, self.head_token_tt, memory_config=mem)  # (N, S, 256)
        x = ttnn.add(contrib, x_base, memory_config=mem)                    # X_base broadcast over N
        ttnn.deallocate(contrib)
        if cap:
            cap("after_head_conditioning", x, rows=self.num_patches, first=True)
        for gp in self.gaze_block_params:
            x = _fused_gaze_block(x, gp, rt, num_heads=8)
        if cap:
            cap("after_gaze_blocks", x, order="fused", first=True)

        if cfg.head == "legacy":
            h = ttnn.linear(x, self.inout_fc1_w, bias=self.inout_fc1_b, activation="relu", memory_config=mem)
            h = ttnn.linear(h, self.inout_fc2_w, bias=self.inout_fc2_b, memory_config=mem)
            io = ttnn.sigmoid(h, memory_config=mem)                                          # (N, S, 1)
            hm = ttnn.linear(x, self.heatmap_w, bias=None, core_grid=_CORE_GRID, memory_config=mem)
            hm = ttnn.add(hm, self.heatmap_b_tt, memory_config=mem)
            hm = ttnn.sigmoid(hm, memory_config=mem)                                         # (N, S, 4)
            return [hm, io]

        h = ttnn.linear(x, self.inout_fc1_w, bias=self.inout_fc1_b, activation="relu", memory_config=mem)  # (N, S, 128)
        act = rt.sigmoid_act if cfg.head == "fused" else None
        out = ttnn.experimental.minimal_matmul(
            [x, h], self.head_w, bias_tensor=self.head_b, fused_activation=act,
            memory_config=mem, compute_kernel_config=rt.head_compute,
        )  # (N, S, 5): K-concat 256 + 128 == 384 padded
        ttnn.deallocate(h)
        if act is None:
            out = ttnn.sigmoid(out, memory_config=mem)
        return [out]

    # ------------------------------------------------------------------ trace plumbing
    def _ensure_input_buffer(self):
        if self._in_dev is None:
            self._in_dev = ttnn.from_torch(
                self._im2col_buf, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def _ensure_scene_trace(self):
        if self._scene_trace is not None:
            return
        dev = self.device
        self._ensure_input_buffer()
        # Warm eager run: compiles every program so the capture only records dispatch.
        xb = self._fused_scene_graph(self._in_dev)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(xb)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            self._x_base = self._fused_scene_graph(self._in_dev)
        except Exception:
            # A capture left open hangs device close: end + release it before propagating.
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.release_trace(dev, tid)
            raise
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        self._scene_trace = tid

    def _ensure_decoder_trace(self, n: int):
        if n in self._dec:
            return
        self._ensure_scene_trace()
        dev = self.device
        mask_dev = ttnn.from_torch(
            torch.zeros(n, self.seq_len, 1, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        outs = self._fused_decoder_graph(mask_dev, self._x_base)
        ttnn.synchronize_device(dev)
        for o in outs:
            ttnn.deallocate(o)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            outs = self._fused_decoder_graph(mask_dev, self._x_base)
        except Exception:
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.release_trace(dev, tid)
            raise
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        self._dec[n] = {"trace": tid, "mask": mask_dev, "outs": outs}

    @property
    def trace_buckets(self) -> List[int]:
        return sorted(self._dec)

    def warmup(self, heads: Optional[Sequence[int]] = None) -> None:
        """Compile + capture everything the server needs BEFORE it reports READY.

        Trace mode: the scene trace and one decoder trace per bucket in ``heads`` (default
        ``TT_FUSED_TRACE_HEADS``). Eager mode: one eager forward (JIT). Idempotent.
        """
        if not self.fused:
            return
        if not self.fused_cfg.trace:
            dummy = torch.zeros(1, 3, self.img_size, self.img_size, dtype=torch.float32)
            self._run_eager(dummy, [WARMUP_BBOX])
            return
        self._ensure_scene_trace()
        for n in sorted(set(int(h) for h in (heads or self.fused_cfg.trace_heads))):
            self._ensure_decoder_trace(n)

    def release_traces(self) -> None:
        """Release captured traces (call before closing the device); buffers stay until GC."""
        if not self.fused or not self.fused_cfg.trace:
            return
        for d in self._dec.values():
            ttnn.release_trace(self.device, d["trace"])
        self._dec.clear()
        if self._scene_trace is not None:
            ttnn.release_trace(self.device, self._scene_trace)
            self._scene_trace = None
            self._x_base = None

    def fused_info(self) -> dict:
        info = dict(self.fused_cfg.as_dict())
        info.update(path=self.path, trace_buckets=self.trace_buckets, scene_traced=self._scene_trace is not None)
        return info

    # ------------------------------------------------------------------ execution
    def _run_traced(self, images, bboxes):
        n = len(bboxes)
        bucket = pick_bucket(n, self.trace_buckets)
        if bucket is None:
            bucket = n  # not warmed for this many heads: capture lazily (seconds, once)
        self._ensure_decoder_trace(bucket)
        dev = self.device
        d = self._dec[bucket]
        # 1. im2col upload  2. scene trace  3. mask upload  4. decoder trace  5. readback (see class doc)
        ttnn.copy_host_to_device_tensor(self._host_input(images), self._in_dev, cq_id=0)
        ttnn.execute_trace(dev, self._scene_trace, cq_id=0, blocking=False)
        ttnn.copy_host_to_device_tensor(self._host_mask(pad_bboxes(bboxes, bucket)), d["mask"], cq_id=0)
        ttnn.execute_trace(dev, d["trace"], cq_id=0, blocking=False)
        return [ttnn.to_torch(o).to(torch.float32) for o in d["outs"]]

    def _run_eager(self, images, bboxes, cap=None):
        """Same fused graph without a trace (TT_FUSED_EAGER=1, ``captures=``, warm-up)."""
        dev = self.device
        in_dev = ttnn.to_device(self._host_input(images), dev)
        x_base = self._fused_scene_graph(in_dev, cap)
        ttnn.deallocate(in_dev)
        mask_dev = ttnn.to_device(self._host_mask(bboxes), dev)
        if cap:
            cap("head_map", mask_dev, rows=self.num_patches, first=True)
        outs = self._fused_decoder_graph(mask_dev, x_base, cap)
        res = [ttnn.to_torch(o).to(torch.float32) for o in outs]
        for o in outs:
            ttnn.deallocate(o)
        ttnn.deallocate(mask_dev)
        ttnn.deallocate(x_base)
        return res

    def _call_fused(self, images, bboxes, captures=None):
        n = len(bboxes)
        cfg = self.fused_cfg

        def cap(key, tt, order=None, rows=None, first=False):
            """Record a stage under its LEGACY name/shape so test_relative_pcc thresholds apply:
            fused-order sequences are rotated back (special token to row 0), patch-only stages are
            cut to the patch rows, batched stages to the first head."""
            t = ttnn.to_torch(tt).to(torch.float32)
            if first:
                t = t[:1]
            if rows is not None:
                t = t[:, :rows]
            if order == "fused":
                t = fused_to_legacy_order(t)
            captures[key] = t

        if captures is not None or not cfg.trace:
            outs = self._run_eager(images, bboxes, cap if captures is not None else None)
        else:
            outs = self._run_traced(images, bboxes)

        fh, fw = self.featmap_h, self.featmap_w
        if cfg.head == "legacy":
            heatmap, inout = decode_legacy_head_outputs(outs[0], outs[1], fh, fw, n, self.out_size)
        else:
            heatmap, inout = decode_head_output(outs[0], fh, fw, n, self.out_size)
        if captures is not None:
            captures["heatmap_compact"] = outs[0][:1, : fh * fw, :4]
            captures["inout_scalar"] = inout[:1]
            captures["heatmap"] = heatmap[:1]
        return {"heatmap": heatmap, "inout": inout}


class TtGazeLLE(_FusedMixin):
    """Gaze-LLE inference entirely on a single Blackhole p150a chip (see module docstring)."""

    def __init__(self, ref_model, device, inout: bool = True):
        self.device = device
        self.inout = inout
        self.ref = ref_model
        backbone = ref_model.backbone
        self.cfg = backbone.cfg
        self.num_heads = self.cfg.num_heads
        self.embed_dim = self.cfg.embed_dim
        self.num_reg_tokens = self.cfg.num_register_tokens
        self.patch_size = backbone.patch_size
        self.img_size = backbone.img_size
        self.num_patches_side = self.img_size // self.patch_size
        self.num_patches = self.num_patches_side ** 2
        self.dim = ref_model.dim  # decoder hidden 256
        self.featmap_h = ref_model.featmap_h
        self.featmap_w = ref_model.featmap_w
        self.out_size = ref_model.out_size

        # TT_FUSED knob, read exactly once here. Default (unset/1) => fused; TT_FUSED=0 => the legacy path below, untouched.
        self.fused_cfg = FusedConfig.from_env()
        self.path = select_forward_path(self.fused_cfg, inout=inout, num_register_tokens=self.num_reg_tokens)
        self.fused = self.path == "fused"
        if self.fused:
            self._init_fused(ref_model, device)
            return

        self.block_params = [_BlockParams(blk, device) for blk in backbone.blocks]
        self.final_norm_w = _to_device(backbone.norm.weight.unsqueeze(0), device)
        self.final_norm_b = _to_device(backbone.norm.bias.unsqueeze(0), device)

        # --- Patch embed as a single on-device matmul. The equivalent torch op is
        # Conv2d(3, embed_dim, k=patch_size, s=patch_size, padding=0). We flatten each
        # patch to (patch_size*patch_size*3) features and run one (588, 768) matmul.
        # Host pre-reshapes the input tensor (pure layout, not inference compute).
        pe_w = backbone.patch_embed_proj.weight.detach()  # (embed_dim, 3, ps, ps)
        w_flat = pe_w.permute(2, 3, 1, 0).reshape(-1, self.embed_dim).contiguous()  # (ps*ps*3, embed_dim)
        self.patch_embed_w = _to_device(w_flat, device, dtype=ttnn.bfloat8_b)
        self.patch_embed_b = _to_device(backbone.patch_embed_proj.bias.detach().unsqueeze(0), device)

        # --- Pre-composed [CLS + pos_cls, REG] prefix + standalone pos_patches.
        cls_tok = backbone.cls_token.detach()
        reg_tok = backbone.reg_token.detach()
        pos_cls = backbone.pos_embed[:, :1].detach()
        pos_patches = backbone.pos_embed[:, 1:].detach().contiguous()
        prefix = torch.cat([cls_tok + pos_cls, reg_tok], dim=1).contiguous()
        self.prefix_tt = _to_device(prefix, device)
        self.pos_patches_tt = _to_device(pos_patches, device)

        # --- Gaze decoder 1x1 projection (768 -> 256) and on-device pos_embed + head_token.
        linear_w = ref_model.linear.weight.squeeze(-1).squeeze(-1).T.contiguous()
        self.proj_w = _to_device(linear_w, device)
        self.proj_b = _to_device(ref_model.linear.bias.unsqueeze(0), device)

        pe = ref_model.pos_embed.permute(1, 2, 0).reshape(1, -1, self.dim).contiguous()
        self.gaze_pos_embed_tt = _to_device(pe, device)
        self.head_token_tt = _to_device(ref_model.head_token.weight.unsqueeze(0), device)  # (1, 1, 256)

        # --- Constants for on-device head-map generation from a 4-scalar bbox.
        fh, fw = self.featmap_h, self.featmap_w
        self.idx_h_tt = _to_device(
            torch.arange(fh, dtype=torch.float32).view(1, fh, 1), device
        )
        self.idx_w_tt = _to_device(
            torch.arange(fw, dtype=torch.float32).view(1, 1, fw), device
        )

        if inout:
            self.inout_token = _to_device(ref_model.inout_token.weight.unsqueeze(0), device)
            # inout_head = Linear 256->128 [0], ReLU [1], Dropout [2], Linear 128->1 [3], Sigmoid [4].
            self.inout_fc1_w = _to_device(ref_model.inout_head[0].weight.T.contiguous(), device)
            self.inout_fc1_b = _to_device(ref_model.inout_head[0].bias.unsqueeze(0), device)
            self.inout_fc2_w = _to_device(ref_model.inout_head[3].weight.T.contiguous(), device)
            self.inout_fc2_b = _to_device(ref_model.inout_head[3].bias.unsqueeze(0), device)

        # --- Fused heatmap head. ConvTranspose2d(k=2, s=2) bias + Conv2d(256->1, k=1, bias=False)
        # is algebraically equivalent to a single per-pixel matmul with weight shape
        # (256, 2, 2) and a scalar bias. We emit a (256, 4) matmul and reshape afterwards.
        ct_w = ref_model.heatmap_head[0].weight.detach()  # (in=256, out=256, kH=2, kW=2)
        ct_b = ref_model.heatmap_head[0].bias.detach()
        c1_w = ref_model.heatmap_head[1].weight.detach().squeeze(-1).squeeze(-1)  # (1, 256)
        w_fused = torch.einsum('ko,ioab->ikab', c1_w, ct_w).squeeze(1)  # (256, 2, 2)
        b_fused = (c1_w @ ct_b).squeeze().item()  # scalar
        w_fused_2d = w_fused.reshape(self.dim, 4).contiguous()
        self.heatmap_w = _to_device(w_fused_2d, device)
        # Broadcast-add scalar bias by uploading a (1, 1, 1) constant.
        self.heatmap_b_tt = _to_device(torch.full((1, 1, 1), float(b_fused)), device)

        self.gaze_block_params = [_GazeBlockParams(blk, device) for blk in ref_model.transformer]

    @staticmethod
    def _reshape_image_for_matmul(images: torch.Tensor, num_patches_side: int, patch_size: int) -> torch.Tensor:
        """(B, 3, H, W) → (B, num_patches, ps*ps*3). Pure layout (permute+reshape)."""
        b = images.shape[0]
        n = num_patches_side
        ps = patch_size
        return (
            images.view(b, 3, n, ps, n, ps)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(b, n * n, ps * ps * 3)
            .contiguous()
        )

    def _encode_scene(self, images: torch.Tensor, capture_fn=None):
        """Run the scene-level portion of the forward once per image.

        Covers: host-reshape → upload → patch-embed matmul → add patch pos_embed
        → concat [CLS+pos_cls, REG] prefix → 12 DINOv2 blocks → final LayerNorm
        → slice off CLS+REG → gaze decoder projection (768→256) → add gaze
        pos_embed. Returns the device tensor holding projected scene features
        of shape ``(1, num_patches, dim)``. Caller owns the deallocate.
        """
        def _cap(key, tt):
            if capture_fn is not None:
                capture_fn(key, tt)

        patches_host = self._reshape_image_for_matmul(images, self.num_patches_side, self.patch_size)
        patches_tt = _to_device(patches_host, self.device)
        patches_tt = ttnn.linear(
            patches_tt, self.patch_embed_w, bias=self.patch_embed_b,
            core_grid=_CORE_GRID, compute_kernel_config=_LOFI,
        )
        _cap("patch_embed", patches_tt)

        patches_tt = ttnn.add(patches_tt, self.pos_patches_tt)
        x_tt = ttnn.concat([self.prefix_tt, patches_tt], dim=1)
        ttnn.deallocate(patches_tt)
        _cap("after_prefix", x_tt)

        capture_blocks = {0, 5, 11} if capture_fn is not None else set()
        for i, bp in enumerate(self.block_params):
            x_tt = _dinov2_block(x_tt, bp, self.num_heads)
            if i in capture_blocks:
                _cap(f"after_block_{i}", x_tt)
        x_tt = ttnn.layer_norm(x_tt, weight=self.final_norm_w, bias=self.final_norm_b, epsilon=1e-6)
        _cap("after_final_norm", x_tt)

        total_prefix = 1 + self.num_reg_tokens
        shp = x_tt.shape
        feat_tt = ttnn.slice(x_tt, [0, total_prefix, 0], [shp[0], shp[1], shp[2]])
        ttnn.deallocate(x_tt)
        _cap("after_slice", feat_tt)

        scene_tt = ttnn.linear(feat_tt, self.proj_w, bias=self.proj_b, core_grid=_CORE_GRID)
        ttnn.deallocate(feat_tt)
        scene_tt = ttnn.add(scene_tt, self.gaze_pos_embed_tt)
        _cap("after_gaze_proj_pos", scene_tt)
        return scene_tt

    def _build_head_contrib(self, bbox, capture_fn=None):
        """Build the ``head_map × head_token`` conditioning tensor for one bbox.

        Returns a (1, num_patches, dim) device tensor. Caller owns deallocate.
        """
        fh, fw = self.featmap_h, self.featmap_w
        xmin_pix = round(bbox[0] * fw)
        ymin_pix = round(bbox[1] * fh)
        xmax_pix = round(bbox[2] * fw)
        ymax_pix = round(bbox[3] * fh)
        h_mask = ttnn.mul(
            ttnn.ge(self.idx_h_tt, float(ymin_pix)),
            ttnn.lt(self.idx_h_tt, float(ymax_pix)),
        )
        w_mask = ttnn.mul(
            ttnn.ge(self.idx_w_tt, float(xmin_pix)),
            ttnn.lt(self.idx_w_tt, float(xmax_pix)),
        )
        mask_2d = ttnn.mul(h_mask, w_mask)
        ttnn.deallocate(h_mask)
        ttnn.deallocate(w_mask)
        head_map_tt = ttnn.reshape(mask_2d, (1, fh * fw, 1))
        if capture_fn is not None:
            capture_fn("head_map", head_map_tt)
        head_contrib = ttnn.mul(head_map_tt, self.head_token_tt)  # broadcast → (1, N, dim)
        ttnn.deallocate(head_map_tt)
        return head_contrib

    def _decode_head(self, scene_tt, bbox, capture_fn=None):
        """Per-head decoder: take the shared scene features, condition on the head
        bbox, run the 3 gaze blocks, and pull the two outputs back to torch.

        ``scene_tt`` is NOT consumed — it is read-only and safe to reuse across
        multiple heads. Returns (heatmap_64x64 torch, inout_scalar torch or None).
        """
        head_contrib = self._build_head_contrib(bbox, capture_fn=capture_fn)
        x_tt = ttnn.add(scene_tt, head_contrib)
        ttnn.deallocate(head_contrib)
        if capture_fn is not None:
            capture_fn("after_head_conditioning", x_tt)

        if self.inout:
            x_tt = ttnn.concat([self.inout_token, x_tt], dim=1)

        for gp in self.gaze_block_params:
            x_tt = _gaze_block(x_tt, gp, num_heads=8)
        if capture_fn is not None:
            capture_fn("after_gaze_blocks", x_tt)

        inout_preds_tt = None
        if self.inout:
            seq = x_tt.shape[1]
            inout_tok = ttnn.slice(x_tt, [0, 0, 0], [1, 1, self.dim])
            patch_out = ttnn.slice(x_tt, [0, 1, 0], [1, seq, self.dim])
            ttnn.deallocate(x_tt)
            h = ttnn.linear(inout_tok, self.inout_fc1_w, bias=self.inout_fc1_b, activation="relu")
            ttnn.deallocate(inout_tok)
            h = ttnn.linear(h, self.inout_fc2_w, bias=self.inout_fc2_b)
            inout_preds_tt = ttnn.sigmoid(h)
        else:
            patch_out = x_tt

        hm = ttnn.linear(patch_out, self.heatmap_w, bias=None, core_grid=_CORE_GRID)
        ttnn.deallocate(patch_out)
        hm = ttnn.add(hm, self.heatmap_b_tt)
        hm = ttnn.sigmoid(hm)
        if capture_fn is not None:
            capture_fn("heatmap_compact", hm)
        return hm, inout_preds_tt

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, bboxes: List[Sequence[float]], captures=None):
        """Run forward for one image with ``N = len(bboxes)`` head bounding boxes.

        The DINOv2 backbone + gaze projection run ONCE; only the bbox-dependent
        tail (``head_map``, 3 gaze blocks, in/out + heatmap heads) runs per head.

        Returns ``{"heatmap": (N, out_h, out_w) torch, "inout": (N,) torch or None}``.
        When ``N == 1`` the shape matches the old single-person contract.

        If ``captures`` is a dict it collects intermediates for the first head;
        scene-level captures cover the shared backbone pass.
        """
        b = images.shape[0]
        assert b == 1, "TtGazeLLE currently supports one image per forward (B=1)"
        assert len(bboxes) >= 1, "need at least one head bbox"

        if self.fused:
            return self._call_fused(images, bboxes, captures)

        def _capture(key, tt_tensor):
            if captures is not None:
                captures[key] = ttnn.to_torch(tt_tensor).to(torch.float32)

        scene_tt = self._encode_scene(images, capture_fn=_capture)

        heatmaps_compact = []
        inout_scalars = []
        try:
            for i, bbox in enumerate(bboxes):
                head_capture = _capture if (captures is not None and i == 0) else None
                hm_tt, inout_tt = self._decode_head(scene_tt, bbox, capture_fn=head_capture)
                heatmaps_compact.append(ttnn.to_torch(hm_tt).to(torch.float32))
                ttnn.deallocate(hm_tt)
                if self.inout and inout_tt is not None:
                    inout_scalars.append(ttnn.to_torch(inout_tt).to(torch.float32).reshape(1))
                    ttnn.deallocate(inout_tt)
        finally:
            ttnn.deallocate(scene_tt)

        # Shape (1, num_patches, 4) per head → fold into (N, featmap_h*2, featmap_w*2).
        stacked = torch.stack([hc[0] for hc in heatmaps_compact], dim=0)  # (N, 1024, 4)
        heatmap = (
            stacked.reshape(-1, self.featmap_h, self.featmap_w, 2, 2)
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, self.featmap_h * 2, self.featmap_w * 2)
        )
        if (self.featmap_h * 2, self.featmap_w * 2) != self.out_size:
            heatmap = F.interpolate(
                heatmap.unsqueeze(1), size=self.out_size, mode="bilinear", align_corners=False
            ).squeeze(1)

        inout_preds = None
        if self.inout and inout_scalars:
            inout_preds = torch.cat(inout_scalars, dim=0)  # (N,)

        if captures is not None:
            captures["heatmap"] = heatmap[:1] if heatmap.shape[0] > 0 else heatmap
            if inout_preds is not None:
                captures["inout_scalar"] = inout_preds[:1]

        return {"heatmap": heatmap, "inout": inout_preds}

        # ---- 9. Download the small outputs.
        heatmap_compact = ttnn.to_torch(hm).to(torch.float32)
        ttnn.deallocate(hm)
        heatmap = (
            heatmap_compact.view(b, self.featmap_h, self.featmap_w, 2, 2)
            .permute(0, 1, 3, 2, 4)
            .reshape(b, self.featmap_h * 2, self.featmap_w * 2)
        )
        if (self.featmap_h * 2, self.featmap_w * 2) != self.out_size:
            heatmap = F.interpolate(
                heatmap.unsqueeze(1), size=self.out_size, mode="bilinear", align_corners=False
            ).squeeze(1)

        inout_preds = None
        if self.inout:
            if captures is not None:
                captures["inout_scalar"] = ttnn.to_torch(inout_preds_tt).to(torch.float32).reshape(b)
            inout_preds = ttnn.to_torch(inout_preds_tt).to(torch.float32).reshape(b)
            ttnn.deallocate(inout_preds_tt)

        if captures is not None:
            captures["heatmap"] = heatmap

        return {"heatmap": heatmap, "inout": inout_preds}
