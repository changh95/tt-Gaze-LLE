# gaze-lle-p150 — Blackhole p150a vs RTX 5090 (same host, same weights, same input)

Date 2026-09-14. Facts only; every GPU number below was measured in this pass, every p150a number is copied
(with its source) from the validation/publish reports and the served probe JSON. The p150a was NOT touched.

## What was run

| | |
|---|---|
| Model | Gaze-LLE (DINOv2 ViT-B/14 @448 backbone, 1025 tokens, + 1x1 proj 768->256 + 3 gaze blocks + heatmap head + in/out MLP), the port's own torch reference `models/gaze-lle-p150/code/gaze_lle/reference/torch_gaze_lle.py::build_gaze_lle("vitb14", inout=True)` + `reference/load_pretrained.py::load_pretrained` — exactly what `server/app.py` builds as `ref` at boot and what the p150a port was PCC-gated against. 89.17 M params. Load report: backbone 174 tensors loaded, pos_embed bicubic 37->32, only `reg_token` missing (expected, the server's `_check_load_report` rule); decoder 45/45, nothing unexpected |
| Weights | `changh95/gaze-lle-weights` @ `f83e53f0f17dd2175f31ceaf9fa650044147ada1` (tt-model.yaml `weights.revision` = `serve.env.TT_WEIGHTS_REVISION`): `dinov2_vitb14_pretrain.pth` (346 MB) + `gazelle_dinov2_vitb14_inout.pt` (12.5 MB) from the HF cache `~/.cache/huggingface/hub/models--changh95--gaze-lle-weights/snapshots/f83e53f0…/`, `hf_hub_download(local_files_only=True)`, `HF_HUB_OFFLINE=1` |
| Input | The served probe's two requests (`logs/megakernel-validate/gaze-lle/serve_probe.py`, the run behind the p150a numbers): **N=1** `media/source_2.png` 640x514 + head bbox `[120,40,190,120]` px; **N=3** `media/source_1.png` 500x334 + `[[116,28,176,104],[269,37,301,81],[87,57,107,76]]` px. Preprocess = `server/app.py::_preprocess` copied verbatim (PIL BILINEAR squash to 448x448, /255, ImageNet mean/std -> `pixel_values [1,3,448,448]` fp32) and `_normalise_bboxes` (px -> fractions of the original image). Batch 1 image |
| GPU | NVIDIA GeForce RTX 5090 (sm_120), driver 580.126.18, power limit 600 W, 32607 MiB; idle 30.9 W |
| venv | `/home/deepgadget/experiments/tt-models/.venv-gpu/main` — Python 3.12.13, torch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19, torchvision 0.26.0, numpy 1.26.4, pillow 12.3.0, huggingface_hub 1.31.0, triton 3.6.0 |
| Scripts | `logs/gpu-vs-p150/gaze-lle/bench_gaze_lle_gpu.py` (uses `logs/gpu-vs-p150/bench_common.py`); `compile_split_addon.py` (re-run of the two `split` torch.compile variants, see below); logs `full_run.log`, `compile_split_addon.log`, `compile_split_addon2.log`, `smoke.log`; raw JSON `reports/gpu-vs-p150/gaze-lle.json` (= `logs/gpu-vs-p150/gaze-lle/result.json`); CPU reference tensors `logs/gpu-vs-p150/gaze-lle/cpu_fp32_reference.pt` |
| Commands | `HF_HUB_OFFLINE=1 .venv-gpu/main/bin/python bench_gaze_lle_gpu.py --iters 50 --warmup 10`, then `… compile_split_addon.py` (twice; the first attempt's reduce-overhead entries failed, see compile table) |
| Loop | per precision and variant: 10 warm-ups + 50 timed iterations, `torch.cuda.synchronize()` before/after each; wall-clock (perf_counter) is the primary number, CUDA-event time recorded alongside (within 0.02 ms of wall); power = `nvidia-smi` sampled every 200 ms during the timed loop (the loop is padded with untimed identical calls to a 2 s window) |
| p150a source | `reports/gpu-vs-p150/p150_numbers.json` -> `reports/megakernel/PUBLISH_SUMMARY.md:16` (Hub `tt serve`: `timing_ms.inference` **5.69 ms** N=1, **6.36 ms** N=3); per-stage split + served totals from `models/gaze-lle-p150/DEVICE_VALIDATION.md` "## Results" (served A/B table, pre-publish shipped image: 5.67 / 16.6 total N=1, 6.38 / 14.8 N=3) and its raw probe `logs/megakernel-validate/gaze-lle/serve_fused_probe.json` (medians 5.67 / **16.635** N=1, 6.38 / **14.765** N=3, 30 warm requests each, `heatmap_format=png`) |

Timing definitions (they match the p150a `timing_ms` keys):

- **incl_h2d** = `pixel_values.to("cuda")` + reference forward + `heatmap.cpu()` + `inout.cpu()`, pageable host tensors (what the server preprocess produces). Compare with p150a `timing_ms.inference` = `model(pixel_values, bboxes_norm)` on the fused path: RM upload of the image + traced DINOv2 scene + decoder trace bucket for N heads + one readback of heatmaps and in/out (**5.69 ms** N=1 / **6.36 ms** N=3, PUBLISH_SUMMARY.md:16).
- **excl_h2d** = forward only, image already resident, outputs left on the device. Note: the reference forward builds the (N,32,32) head mask on the host from the bboxes on every call and `.to(device)`s it inside `forward`; this small host op + upload is inside BOTH variants (the p150a fused path also builds the mask on the host and uploads it).
- **served-like** = base64 decode + PNG decode (PIL) + `_normalise_bboxes` + `_preprocess` + incl_h2d forward + the server's prediction loop (argmax -> gaze_target, 64x64 uint8 PNG encode per head, `heatmap_format=png` as the probe sent). Compare with p150a `timing_ms.total` (**16.635 ms** N=1, **14.765 ms** N=3). `p150_numbers.json` says "npz encode" for this total; the probe actually sent `png`, so png is what is matched here.
- **N=3**: the torch reference is a "one bbox per image" API (`assert len(bboxes) == B`), so three heads on one image are timed two ways: **batch-reference** = `model(pixel_values.repeat(3,1,1,1), 3 bboxes)` (backbone runs 3x, 3 images uploaded — the form the port's own validation compared against: "3 single-head torch forwards"), and **shared-backbone** = backbone once + decoder batched over the 3 heads (`bench_gaze_lle_gpu.py::gaze_tail`, the same statements as `GazeLLE.forward` after `self.backbone`; it is what the p150a fused pipeline does). Shared-backbone == batch-reference bit-for-bit on CPU fp32 (PCC 1.0, max abs diff 0.0) and PCC 1.000000 on the GPU in fp32 strict. The shared-backbone row is the like-for-like comparison with the p150a's one-image N-head request.

## Correctness check (GPU vs CPU fp32 reference)

CPU fp32 single forward: 382 ms (N=1), 1247 ms (N=3 batch). GPU fp32 strict (no TF32), same `pixel_values` / bboxes:

| case | heatmap PCC (N,64,64) | max abs diff heatmap | in/out GPU vs CPU | argmax row/col (GPU = CPU) | p150a served argmax (probe) |
|---|---:|---:|---|---|---|
| N=1 source_2 | **1.000000** | 6.8e-6 | 0.9602 vs 0.9602 (diff 1.2e-7) | [49, 45] | [49, 45], inout 0.969 |
| N=3 batch-reference | **1.000000** (per head 1.0 / 1.0 / 1.0) | 8.0e-6 | [0.0269, 0.7694, 0.6322], diff 3.6e-6 | [33,37] [19,38] [12,12] | [33,37] [19,38] **[32,36]**, inout 0.025 / 0.777 / 0.641 |
| N=3 shared-backbone | **1.000000** (1.0 / 1.0 / 1.0) | 9.8e-6 | same | same | |

PCC > 0.999 holds for fp32; the GPU runs the right model, and the fp32 argmax matches the p150a's served prediction for
N=1 and for heads 0 and 1 of N=3. Fact about head 2 (the small face at the back of source_1): the fp32 reference (CPU and
GPU) peaks at row 12 / col 12 while the p150a (bf16) served row 32 / col 36; the GPU under **bf16 autocast** also gives
[32, 36] (head-2 PCC vs fp32 0.98699), tf32 and fp16 stay at [12, 12] — that head's map has two near-equal peaks and
bf16 rounding flips the argmax on both accelerators. (DEVICE_VALIDATION.md "Results" documents the p150a's head-1 = [19, 38]
agreeing with torch; head 2 was not compared there.)

Per-precision accuracy vs the CPU fp32 reference (heatmap PCC; the p150a's own gate value on the same metric was 0.9984 vs torch, VS:36):

| GPU precision | N=1 PCC / max abs diff / inout | N=1 argmax | N=3 shared PCC (per head) | N=3 argmax vs p150a |
|---|---|---|---|---|
| fp32 strict | 1.000000 / 6.8e-6 / 0.9602 | [49,45] | 1.000000 (1.0, 1.0, 1.0) | heads 0,1 same; head 2 [12,12] vs [32,36] |
| tf32 | 1.000000 / 3.8e-4 / 0.9602 | [49,45] | 0.999999 (0.999999, 1.0, 0.999998) | same as fp32 |
| bf16 autocast | 0.999607 / 2.0e-2 / 0.9570 | [49,45] | 0.995843 (0.998528, 0.996848, 0.986990) | all 3 heads == p150a ([33,37] [19,38] [32,36]) |
| fp16 autocast | 0.999894 / 8.5e-3 / 0.9604 | [49,45] | 0.999565 (0.999894, 0.999882, 0.998079) | same as fp32 |
| p150a (bf16 device, VS:36) | heatmap PCC 0.9984 vs torch; GazeFollow AUC 0.9540 (torch 0.9543) | [49,45] | | |

fp16 autocast is numerically fine (PCC > 0.999 on both cases, no overflow), so it is timed; bf16 autocast is the p150a's precision class.

## GPU latency, N=1 (batch 1, 448x448, one head; median / min / p90 of 50 iterations, wall-clock ms)

Eager PyTorch:

| precision | incl_h2d median / min / p90 | excl_h2d median / min / p90 | CUDA-event excl | first call ms | power mean W (excl loop / incl loop) | GPU util % (excl) | peak mem alloc / reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | **9.188** / 9.129 / 9.240 | **9.053** / 8.947 / 9.076 | 9.044 | 9.0 (183.7 on the very first CUDA call incl. cuDNN init) | 532.6 / 359.3 | 96.6 | 468 / 810 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | **5.721** / 5.697 / 5.741 | **5.510** / 5.488 / 5.542 | 5.501 | 5.5 | 498.7 / 514.4 | 94.2 | 468 / 810 |
| bf16 autocast (+TF32 remainder) | **4.996** / 4.963 / 5.014 | **4.844** / 4.814 / 5.103 | 4.831 | 5.1 | 476.9 / 452.9 | 89.3 | 486 / 810 |
| fp16 autocast (+TF32 remainder) | **5.786** / 5.735 / 5.815 | **5.598** / 5.536 / 5.623 | 5.585 | 5.8 | 456.7 / 441.1 | 91.7 | 440 / 810 |
| tf32, pinned host input (informational) | 5.679 / 5.661 / 5.697 | | | | | | |
| bf16 autocast, pinned host input (informational) | 4.960 / 4.928 / 5.020 | | | | | | |

`torch.compile` (inductor, `dynamic=False`; every compile well under the 5-min budget). `whole` = `torch.compile(model)`
(the head mask is built on the host and `.to(cuda)`d inside the compiled forward); `split` = compiled backbone + compiled
`gaze_tail` with the mask built eagerly and uploaded before the call, `torch.compiler.cudagraph_mark_step_begin()` per
forward (without it the two CUDA-graph functions fail with "accessing tensor output of CUDAGraphs that has been
overwritten", `compile_split_addon.log`). `split` compile times of ~1 s are inductor cache hits from the first attempt.

| variant | compile s | incl_h2d median / min / p90 | excl_h2d median / min / p90 | power W (excl) | peak mem MiB | heatmap PCC vs CPU fp32 | argmax / inout |
|---|---:|---:|---:|---:|---:|---:|---|
| tf32 + compile default, whole | 7.1 | 5.623 / 5.609 / 5.702 | 5.401 / 5.385 / 5.431 | 469.3 | 462 | 0.999997 | [49,45] / 0.9603 |
| tf32 + compile reduce-overhead (CUDA graphs), whole | 4.9 | **5.302** / 5.278 / 5.401 | **5.118** / 5.105 / 5.129 | 490.9 | 344 | 0.999997 | [49,45] / 0.9603 |
| tf32 + compile reduce-overhead, split | 1.5 | 5.332 / 5.315 / 5.410 | 5.168 / 5.154 / 5.184 | 477.6 | 344 | 0.999997 | [49,45] / 0.9603 |
| bf16 autocast + compile default, whole | 7.5 | 4.171 / 4.155 / 4.185 | 3.975 / 3.964 / 4.025 | 503.3 | 411 | 0.999604 | [49,45] / 0.957 |
| bf16 autocast + compile default, split | 7.0 | 4.307 / 4.288 / 4.323 | 4.105 / 4.084 / 4.115 | 488.0 | 411 | 0.999592 | [49,45] / 0.957 |
| bf16 autocast + compile reduce-overhead, whole | 6.5 | **3.754** / 3.646 / 3.796 | **3.463** / 3.450 / 3.490 | 555.7 | 344 | 0.997612 (max abs diff 0.072) | [49,45] / 0.957 |
| bf16 autocast + compile reduce-overhead, split | 0.8 | 3.805 / 3.784 / 3.820 | 3.515 / 3.506 / 3.525 | 536.7 | 344 | 0.999540 | [49,45] / 0.957 |

Other facts: build + `torch.load` of both checkpoints + `load_state_dict` 0.49 s on the CPU, fp32 state -> cuda 0.052 s;
first fp32 CUDA call 184 ms (CUDA/cuDNN init). GPU utilisation during the eager excl loops is 89-97 % — a 1025-token
ViT-B at batch 1 keeps the RTX 5090 busy (unlike rf-detr's launch-bound graph), so strict fp32 (no TF32) costs 1.6x tf32
and CUDA graphs buy only 7 % (tf32) to 25 % (bf16). fp16 autocast is slower than bf16 here (5.79 vs 5.00 ms; the reference
attention is explicit matmul + softmax, not SDPA). The whole-model bf16 CUDA-graph variant is the fastest but its PCC drops
to 0.9976 (inductor fusions under autocast with the in-graph mask upload); the split form keeps 0.9995 at +0.05 ms.

## GPU latency, N=3 (one 448x448 image, three heads; median / min / p90 of 50, wall-clock ms)

| precision | shared-backbone incl_h2d | shared-backbone excl_h2d | power W / peak MiB | batch-reference incl_h2d (image x3) | batch-reference excl_h2d | power W / peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 strict | **10.241** / 10.182 / 10.275 | 10.034 / 9.977 / 10.103 | 544.2 / 567 | 21.930 / 21.817 / 22.128 | 21.594 / 21.428 / 21.687 | 595.3 / 700 |
| tf32 | **6.658** / 6.630 / 6.679 | 6.456 / 6.429 / 6.477 | 509.3 / 567 | 15.514 / 15.474 / 15.746 | 15.132 / 15.094 / 15.222 | 530.8 / 700 |
| bf16 autocast | **5.895** / 5.867 / 5.914 | 5.720 / 5.693 / 5.735 | 487.8 / 611 | 14.167 / 14.126 / 14.211 | 13.771 / 13.736 / 13.876 | 508.4 / 754 |
| fp16 autocast | **6.936** / 6.879 / 6.970 | 6.736 / 6.690 / 6.772 | 466.7 / 518 | 15.843 / 15.793 / 15.936 | 15.438 / 15.419 / 15.466 | 479.4 / 619 |

Going from 1 to 3 heads costs the GPU +0.9 ms (bf16) / +0.9 ms (tf32) / +1.05 ms (fp32) on the shared-backbone form; the
p150a's decoder bucket costs +0.67 ms (5.69 -> 6.36). The batch-reference form triples the backbone and is 2.2-2.4x slower
than shared-backbone; it is listed because it is literally the port's reference API, not as the comparison row.

## Served-like loop (same host work as `server/app.py::predict`, 50 iterations, medians ms)

| case | GPU precision | decode + bbox + preprocess (server `preprocess`) | inference (incl_h2d) | postprocess + PNG encode (server `encode`) | **total** | p90 total | power W | predictions |
|---|---|---:|---:|---:|---:|---:|---:|---|
| N=1 640x514 | fp32 strict | 9.52 | 9.21 | 0.47 | **19.193** | 19.243 | 356 | [49,45] 0.960 |
| | tf32 | 8.67 | 5.71 | 0.48 | **14.858** | 15.048 | 269 | [49,45] 0.960 |
| | bf16 autocast | 9.54 | 5.05 | 0.46 | **15.054** | 15.122 | 224 | [49,45] 0.957 |
| | fp16 autocast | 9.53 | 5.79 | 0.45 | **15.768** | 15.960 | 215 | [49,45] 0.960 |
| | p150a (serve_fused_probe.json, 30 warm) | ~11.0 (= total - inference) | 5.67 | | **16.635** | max 17.53 | not measured | [49,45] 0.969 |
| N=3 500x334 | fp32 strict | 6.43 | 10.21 (shared-backbone) | 1.51 | **18.132** | 18.275 | 305 | [33,37] [19,38] [12,12] |
| | tf32 | 5.57 | 6.63 | 1.49 | **13.692** | 14.556 | 309 | same |
| | bf16 autocast | 6.42 | 5.92 | 1.44 | **13.766** | 13.816 | 274 | [33,37] [19,38] [32,36] (== p150a) |
| | fp16 autocast | 5.57 | 6.96 | 1.49 | **14.025** | 14.908 | 268 | as fp32 |
| | p150a (serve_fused_probe.json) | ~8.4 (= total - inference) | 6.38 | | **14.765** | max 15.1 | not measured | [33,37] [19,38] [32,36] |

The host stages are the same code on both sides (PIL PNG decode + 448 resize ~5.6-9.5 ms, 0.5 ms per PNG-encoded heatmap),
so the served totals differ almost only by the inference term. The GPU host stages here ran ~1.5-2 ms faster than the
p150a container's (different process, no uvicorn/pydantic), which slightly favours the GPU e2e rows.

## Comparison with the p150a (matching definitions)

Ratio = p150a ms / GPU ms (> 1 means the GPU is faster). p150a precision: bf16 activations on device, LoFi fused
matmuls (dit), SDPA q128/k128 exact-exp, one scene trace + one decoder trace per head bucket (`TT_FUSED=1` default,
DEVICE_VALIDATION.md "Results"). Device-forward rows use the Hub `tt serve` medians (5.69 / 6.36 ms).

| row | p150a (definition) | GPU precision | GPU ms | ratio p150a/GPU |
|---|---:|---|---:|---:|
| device forward N=1 (p150a `timing_ms.inference` incl. upload + readback vs GPU incl_h2d) | 5.69 (PUBLISH_SUMMARY.md:16, Hub tt serve; probe 5.67) | fp32 strict | 9.188 | **0.62** |
| | | tf32 | 5.721 | **0.99** |
| | | bf16 autocast | 4.996 | **1.14** |
| | | fp16 autocast | 5.786 | **0.98** |
| | | tf32 + compile reduce-overhead (whole) | 5.302 | **1.07** |
| | | bf16 autocast + compile default (whole) | 4.171 | **1.36** |
| | | bf16 autocast + compile reduce-overhead (split, PCC 0.9995) | 3.805 | **1.50** |
| | | bf16 autocast + compile reduce-overhead (whole, PCC 0.9976) | 3.754 | **1.52** |
| GPU forward only N=1 (excl_h2d) vs the same p150a 5.69 | 5.69 | fp32 strict / tf32 / bf16 / fp16 | 9.053 / 5.510 / 4.844 / 5.598 | 0.63 / 1.03 / 1.17 / 1.02 |
| | | tf32 + CUDA graphs / bf16 + CUDA graphs (whole) | 5.118 / 3.463 | 1.11 / 1.64 |
| device forward N=3, shared backbone (p150a 3-head bucket vs GPU backbone x1 + 3 heads, incl_h2d) | 6.36 (PUBLISH_SUMMARY.md:16; probe 6.38) | fp32 strict | 10.241 | **0.62** |
| | | tf32 | 6.658 | **0.96** |
| | | bf16 autocast | 5.895 | **1.08** |
| | | fp16 autocast | 6.936 | **0.92** |
| device forward N=3, batch-reference (image x3, informational) | 6.36 | fp32 / tf32 / bf16 / fp16 | 21.930 / 15.514 / 14.167 / 15.843 | 0.29 / 0.41 / 0.45 / 0.40 |
| served e2e N=1 (p150a `timing_ms.total` vs GPU served-like total) | 16.635 (serve_fused_probe.json; DEVICE_VALIDATION.md 16.6) | fp32 strict | 19.193 | **0.87** |
| | | tf32 | 14.858 | **1.12** |
| | | bf16 autocast | 15.054 | **1.11** |
| | | fp16 autocast | 15.768 | **1.05** |
| served e2e N=3 | 14.765 (serve_fused_probe.json; DEVICE_VALIDATION.md 14.8) | fp32 strict | 18.132 | **0.81** |
| | | tf32 | 13.692 | **1.08** |
| | | bf16 autocast | 13.766 | **1.07** |
| | | fp16 autocast | 14.025 | **1.05** |

Reading: this is the closest race in the set so far. In the p150a's own precision class (bf16 activations) the eager
RTX 5090 forward is 1.14x faster than the p150a's fused traces at N=1 (5.00 vs 5.69 ms incl. transfers) and 1.08x at
N=3 (5.90 vs 6.36); with TF32 (fp32 tensors) the two are at parity (5.72 vs 5.69, 0.99x); in strict fp32 the GPU is
1.6x SLOWER than the p150a (9.19 vs 5.69 ms). The best GPU configuration with PCC >= 0.999 (bf16 autocast + inductor +
CUDA graphs, split form) is 1.50x faster than the p150a device forward (3.81 vs 5.69 ms). End-to-end, the shared
~8-11 ms of host PNG decode / resize / PNG-encode work compresses the gap to 1.05-1.12x (GPU faster) for tf32/bf16/fp16
and 0.81-0.87x (p150a faster) for strict fp32.

Not measured / not claimed: p150a power (not measured in any pass -> no power or efficiency comparison; the GPU drew
457-596 W mean during the dense excl loops, 215-360 W during the served-like loops, 31 W idle). p150a numbers were not
re-measured. Neither side's number includes HTTP/JSON framing. The p150a's N=3 served total (14.765) is lower than its N=1
total (16.635) because source_1.png is a smaller image (500x334, 226 KB vs 640x514, 527 KB) — the same ordering shows on the
GPU side (13.7 vs 14.9 ms).

## Reproduce

```bash
cd /home/deepgadget/experiments/tt-models/logs/gpu-vs-p150/gaze-lle
HF_HUB_OFFLINE=1 /home/deepgadget/experiments/tt-models/.venv-gpu/main/bin/python bench_gaze_lle_gpu.py --iters 50 --warmup 10 | tee full_run.log
HF_HUB_OFFLINE=1 /home/deepgadget/experiments/tt-models/.venv-gpu/main/bin/python compile_split_addon.py | tee compile_split_addon.log   # merges compile[*+split]
# outputs: /home/deepgadget/experiments/tt-models/reports/gpu-vs-p150/gaze-lle.json, ./result.json, ./cpu_fp32_reference.pt
```

GPU released after the run: `nvidia-smi --query-compute-apps=pid --format=csv,noheader` -> empty (29 W, 668 MiB display only).
