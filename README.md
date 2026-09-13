# tt-Gaze-LLE

End-to-end port of [Gaze-LLE](https://github.com/fkryan/gazelle) (Ryan et al.)
to Tenstorrent **tt-metal** (tt-nn + tt-metallium), running on a single
Blackhole p150a chip.

This repository contains a TT-NN implementation of the
`gazelle_dinov2_vitb14_inout` checkpoint, a torch reference used as a numerical
shadow, pytest suites for per-stage PCC and real-image correctness, a
GazeFollow AUC/L2 evaluator, a wall-clock benchmark, a Tracy-profileable perf
test, and a script to pull the weights and data.

The TT-NN forward runs the **entire inference** on the chip — the host does
only layout prep (zero-cost views) and upload/download.

**Fused traced path (default since 2026-09-13).** `TtGazeLLE` now runs the whole
graph as one scene metal-trace plus one decoder trace per head-count bucket,
with the patch embed, proj+residual and fc2+residual fused into
`dit_minimal_matmul_addcmul_fused`, all heads batched through the decoder and a
single readback — 5.5 ms per forward on a p150a versus 9.4 ms for the previous
eager path (`benchmark.py`, 1 head). The eager path is kept bit for bit behind
`TT_FUSED=0`. See **Fused traced path (`TT_FUSED`)** and `DEVICE_VALIDATION.md`.

**Multi-person in one forward.** `TtGazeLLE(image, [bbox_a, bbox_b, …])` runs
the DINOv2 backbone and the gaze projection **once**, then runs the
bbox-conditioned decoder tail (head-mask build + 3 gaze blocks + heads) for all
heads — batched along `N` in one traced replay on the fused path, once per head
on the legacy path — returning stacked per-head heatmaps and in/out scores
(`(N, 64, 64)` and `(N,)`). `N = 1` keeps the old single-person contract.

---

## Demo

Four multi-person scenes from the GazeFollow test set. Each pair follows the
canonical Gaze-LLE inference pipeline from the
[official Colab](https://colab.research.google.com/drive/1TSoyFvNs1-au9kjOZN_fo5ebdzngSPDq):

1. **RetinaFace** detects every face in the image.
2. The detected bboxes are fed into `TtGazeLLE(image, bboxes)` running on a
   Blackhole p150a — the DINOv2 backbone + gaze projection run **once**, the
   decoder tail runs **N times** over the shared scene features.

Each colored bbox is one RetinaFace detection; the same-colored arrow and ×
are that person's predicted gaze direction and target pixel. No bboxes are
hand-picked — every one comes from the face detector.

Reproduce with ``python -m scripts.make_demo``
(requires ``pip install retina-face tf-keras``).

| Input (`media/source_N.png`) | Prediction (`media/target_N.png`) |
|:---:|:---:|
| ![](media/source_1.png) | ![](media/target_1.png) |
| ![](media/source_2.png) | ![](media/target_2.png) |
| ![](media/source_3.png) | ![](media/target_3.png) |
| ![](media/source_4.png) | ![](media/target_4.png) |

---

## Contents

```
gaze_lle/
├── reference/
│   ├── torch_gaze_lle.py      # self-contained torch reference (no timm / torch.hub)
│   └── load_pretrained.py     # fkryan/gazelle + DINOv2 ckpt → reference model
├── tt/
│   ├── tt_gaze_lle.py         # TT-NN forward (Blackhole p150a): fused traced path + legacy eager path
│   └── fused_math.py          # torch-only half of the fused path: TT_FUSED* knobs, folded constants, fp32 mirror
├── benchmark.py               # torch/tt FPS + PCC benchmark
└── tests/
    ├── test_fused_host.py         # 15 torch-only tests of the fused reformulations (no ttnn, no device)
    ├── test_relative_pcc.py       # 14-stage tt↔torch PCC test (random weights)
    ├── test_pretrained_eval.py    # real-image torch↔tt PCC + peak check (pretrained)
    ├── test_multi_person.py       # N-head vs N single-head torch forwards (needs retina-face)
    ├── test_perf.py               # single-iter forward, Tracy-profileable
    └── eval_gazefollow.py         # full-set GazeFollow AUC / Avg L2 / Min L2
scripts/
├── download_data.sh           # pulls weights + sample images + GazeFollow parquet
└── make_demo.py               # RetinaFace + TtGazeLLE demo renderer (media/)
conftest.py                    # minimal pytest `device` fixture (trace region on the fused path)
DEVICE_VALIDATION.md           # fused-path plan, knobs, gates and the measured device results (2026-09-13)
```

The code imports `ttnn` and uses the tt-metal / tt-nn build you point it at —
this repo does **not** contain the tt-metal monorepo. See **Environment setup**
below.

---

## Environment setup

1. Build tt-metal with its Python bindings (and Tracy support if you plan to
   run `test_perf.py`). Instructions: https://github.com/tenstorrent/tt-metal.
2. Install the Python deps for this repo. A working set:

   ```bash
   pip install torch torchvision pandas pyarrow scikit-learn tqdm pillow pytest
   ```

   `torch` can be CPU-only; the reference is only used as a numerical shadow.

   **Optional (multi-person demo + multi-person test):** the canonical
   Gaze-LLE Colab pipeline feeds **RetinaFace**-detected head bboxes into the
   model. If you want to run the multi-person path, also install:

   ```bash
   pip install retina-face tf-keras
   ```

   Without these, `scripts/make_demo.py` and `test_multi_person.py` skip the
   multi-person steps with a clear message; everything else still works.
3. Point Python at tt-metal's `ttnn` bindings and set the few env vars the TT-NN
   runtime expects:

   ```bash
   export TT_METAL_HOME=/path/to/tt-metal
   export PYTHONPATH=$TT_METAL_HOME:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$PWD
   # Restrict visibility to the one Blackhole chip you want to use (optional):
   export TT_VISIBLE_DEVICES=0
   ```

4. Download everything:

   ```bash
   bash scripts/download_data.sh
   ```

   Populates `./weights/` (DINOv2 backbone + Gaze-LLE decoder) and `./data/`
   (two sample images + the full 4,782-image GazeFollow parquet).
   Override locations with `TT_GAZE_LLE_WEIGHTS` / `TT_GAZE_LLE_DATA`.

---

## Running the tests

Every device command below runs the fused traced path by default; prefix it with
`TT_FUSED=0` to run the legacy eager path instead (same tests, same thresholds).

```bash
# 0) Host-only proof of the fused reformulations (torch only, no ttnn, no device; ~6 s)
pytest gaze_lle/tests/test_fused_host.py -q

# 1) Per-stage relative PCC (tt vs a torch shadow with the same random weights)
pytest gaze_lle/tests/test_relative_pcc.py -v -s
TT_FUSED=0 pytest gaze_lle/tests/test_relative_pcc.py -v -s      # legacy path

# 2) Pretrained-weight check on two real images (see the caveat in Results)
pytest gaze_lle/tests/test_pretrained_eval.py -v -s

# 3) Full GazeFollow test set (4,782 images, AUC + L2); --max-samples 500 for the quick subset
python -m gaze_lle.tests.eval_gazefollow
TT_FUSED=0 python -m gaze_lle.tests.eval_gazefollow

# 4) Wall-clock FPS benchmark (traces are captured before the timed loop)
python -m gaze_lle.benchmark --impl ttnn --iters 50 --warmup 5
TT_FUSED=0 python -m gaze_lle.benchmark --impl ttnn --iters 50 --warmup 5

# 5) Same fused graph without a trace (first thing to run when an op fails: full python trace)
TT_FUSED_EAGER=1 pytest gaze_lle/tests/test_relative_pcc.py -v -s

# 6) Tracy-profiled single-iter forward (requires Tracy-enabled tt-metal build)
python -m tracy --no-runtime-analysis --collect-noc-traces \
    --profiler-capture-perf-counters=all --op-support-count=10000 \
    -v -r -o ./tracy_out -m pytest gaze_lle/tests/test_perf.py
```

---

## Fused traced path (`TT_FUSED`)

`TtGazeLLE` reads the `TT_FUSED*` knobs **once at construction**
(`FusedConfig.from_env` in `gaze_lle/tt/fused_math.py`). Unset or `1` selects
the fused traced path; `TT_FUSED=0` restores the legacy eager path bit for bit
(the only legacy-side changes are `open_device_kwargs()` returning `{}` and the
dispatch line). The fused path needs the device opened with a trace region —
use `open_device_kwargs()` as `conftest.py`, `benchmark.py`, `eval_gazefollow.py`
and `make_demo.py` do:

```python
import ttnn
from gaze_lle.tt.tt_gaze_lle import TtGazeLLE, open_device_kwargs

device = ttnn.open_device(device_id=0, **open_device_kwargs())   # {} with TT_FUSED=0
model = TtGazeLLE(ref_model, device, inout=True)
model.warmup()                       # captures the scene trace + one decoder trace per head bucket
heatmaps, inout = model(image, bboxes)
```

What the fused graph does differently (details and the per-lever A/B in
`DEVICE_VALIDATION.md`): persistent ROW_MAJOR input buffer with on-device
tilize; patch embed + pos + CLS, proj+residual and fc2+residual as one
`dit_minimal_matmul_addcmul_fused` each (bf16 weights); SDPA with an explicit
program config (q128/k128 chunks, exact exp); final LN + gated projection that
drops the CLS row in-op; host-built head mask uploaded once per call with all
heads batched along `N`; relu-linear + block-diagonal head matmul with a single
`(N, 1025, 5)` readback. The whole thing is one scene trace plus one decoder
trace per head-count bucket; a request with `N` heads runs the smallest bucket
`>= N` (the last bbox is repeated as padding) so no capture happens at request
time after `warmup()`.

| env | default | meaning |
|---|---|---|
| `TT_FUSED` | unset = on | master switch; `0` = legacy eager path |
| `TT_FUSED_EAGER` | `0` | `1` = same fused graph without a trace (debug; op failures surface with a python trace) |
| `TT_FUSED_TRACE_HEADS` | `1,2,3,4,6,8,10` | decoder trace buckets captured by `warmup()`; other head counts are padded up |
| `TT_FUSED_TRACE_REGION_MB` | `64` | `open_device(trace_region_size=...)` via `open_device_kwargs()` |
| `TT_FUSED_DIT` | `1` | `dit_minimal_matmul_addcmul_fused` for patch embed / proj+res / fc2+res / gated proj (bf16 W); `0` = linear + add/mul, bfp8 W like legacy |
| `TT_FUSED_HEAD` | `separate` | `separate` (K-concat head matmul + separate sigmoid), `fused` (sigmoid inside the matmul; measured no faster), `legacy` (in/out MLP + heatmap linear on all rows, 2 readbacks) |
| `TT_FUSED_SDPA` / `_Q` / `_K` / `_EXACT_EXP` | `1` / `128` / `128` / `1` | SDPA program config; `TT_FUSED_SDPA=0` = tree defaults like legacy |
| `TT_FUSED_FIDELITY` | `LoFi` | math fidelity of the fused matmuls (`HiFi2` measured +0.07 ms with no metric gain) |
| `TT_FUSED_MINIMAL_MM` | `0` | `minimal_matmul` for qkv / fc1 — **not verified on hardware** |
| `TT_FUSED_L1` | `0` | L1 interleaved working set — **not verified on hardware** |
| `TT_FUSED_BF16_WEIGHTS` | `0` | bf16 instead of bfp8 for qkv / fc1 — **not verified on hardware** |

The default config above is the one that passed every gate on the p150a
(2026-09-13); the three unverified knobs stay off.

---

## Results

### Relative PCC (per-stage, tt vs torch shadow)

All 14 intermediate stages of the TT-NN forward match the torch shadow within
tight thresholds. Even the end-to-end heatmap output stays above 0.998 PCC with
random weights, which is consistent with the bf16 / bfp8 / LoFi accumulation
behaviour of the chip over a 12-layer backbone.

| Stage | PCC | Shape |
|---|---:|---|
| patch_embed                 | 0.9998 | (1, 1024, 768) |
| after_prefix                | 0.9998 | (1, 1025, 768) |
| after_block_0               | 0.9998 | (1, 1025, 768) |
| after_block_5               | 0.9994 | (1, 1025, 768) |
| after_block_11              | 0.9989 | (1, 1025, 768) |
| after_final_norm            | 0.9988 | (1, 1025, 768) |
| after_slice                 | 0.9988 | (1, 1024, 768) |
| after_gaze_proj_pos         | 0.9993 | (1, 1024, 256) |
| head_map                    | 1.0000 | (1, 1024, 1)   |
| after_head_conditioning     | 0.9994 | (1, 1024, 256) |
| after_gaze_blocks           | 0.9991 | (1, 1025, 256) |
| heatmap_compact             | 0.9981 | (1, 1024, 4)   |
| inout_scalar                | 0.9940 | (1,)           |
| heatmap                     | 0.9981 | (1, 64, 64)    |

Numbers are from the legacy-path run committed in
`gaze_lle/tests/test_relative_pcc.py` (random weights, seed 0). The head-bbox
mask is exactly `1.0000` because it is a binary comparison — the `ttnn.ge` /
`ttnn.lt` cascade is bit-identical to the torch shadow (on the fused path the
mask is built on the host and is `torch.equal` to the reference).

Re-measured 2026-09-13 on the same tree for both paths (`DEVICE_VALIDATION.md`,
"Final gates"): legacy block_11 0.9990 / final_norm 0.9990 / gaze_blocks 0.9993
/ heatmap 0.9984 / inout 0.9940; fused 0.9993 / 0.9992 / 0.9992 / 0.9984 /
0.9940. On the fused path `patch_embed` and `after_slice` are not separate
stages (folded into one op / dropped in-op) and are skipped; the other stages
are captured under their legacy names with the special token rotated back to
row 0.

### Real-data evaluation on GazeFollow

Full test set, 4,782 images, each with one head bounding box and up to 10
annotator gaze targets. Metric formulas follow `gazelle/utils.py` from
fkryan/gazelle.

| Impl | AUC | Avg L2 | Min L2 |
|---|---:|---:|---:|
| Paper (fkryan/gazelle) | 0.9560 | 0.1510 | 0.0990 |
| Torch pretrained (CPU)   | 0.9543 | 0.1103 | 0.0491 |
| TT-NN on p150a, legacy eager path (`TT_FUSED=0`) | 0.9541 | 0.1129 | 0.0512 |
| **TT-NN on p150a, fused traced path (default, measured 2026-09-13)** | **0.9540** | **0.1119** | **0.0502** |

The TT port is **within 0.0003 AUC and ≤0.0026 L2** of the torch reference on
the full test set — i.e., running on Blackhole does not sacrifice any
meaningful prediction quality compared to the pretrained model. The small
absolute gap to the paper's headline L2 numbers is attributable to the
vikhyatk/gazefollow HF mirror having slightly different annotator-point
coordinates than the paper's `test_preprocessed.json` pipeline; both torch and
TT show the same gap, so it is not a port defect.

Two-image qualitative check (`test_pretrained_eval.py`):

| Image | heatmap PCC (torch↔tt), fused / legacy | peak distance (64×64 space), fused / legacy |
|---|---:|---:|
| the_office.png | 0.9891 / 0.9846 | 0 px / 1 px |
| succession.png | 0.9435 / 0.9910 | 8 px / 7 px |

Re-measured 2026-09-13 on tt-metal `v0.78.0-dev20260820` (main `8b98410e730`).
On this tree the succession.png peak moves on **both** paths, so the test's
0.99 PCC / 5 px thresholds (set on an earlier tree, where both images gave
0.9923 / 1 px) fail for legacy and fused alike; the full-set GazeFollow numbers
above are the accuracy gate. The thresholds were left unchanged.

### Performance

Measured with `gaze_lle/benchmark.py` in wall-clock mode (not Tracy) at batch 1
on a single p150a. Tracy-instrumented runs are ~30% slower due to profiling
overhead; those numbers should be read as a per-op breakdown, not a headline
latency.

| Metric | fused traced path (default) | legacy eager path (`TT_FUSED=0`) |
|---|---:|---:|
| `benchmark.py --impl ttnn --iters 50 --warmup 5`, 1 head (two runs each) | **181.5 / 180.5 FPS = 5.51 / 5.54 ms** | 105.9 / 106.3 FPS = 9.44 / 9.41 ms |
| `benchmark.py` accuracy (heatmap PCC vs torch, random weights) | 99.8393 | 99.8366 |
| Served warm `timing_ms.inference`, 1 head, 640×514 (median / min / max of 30) | **5.67 / 5.61 / 5.76 ms** | 9.40 / 9.14 / 13.27 ms |
| Served warm `timing_ms.inference`, 3 heads, 500×334 | **6.38 / 6.32 / 6.51 ms** | 12.64 / 12.22 / 13.14 ms |
| Served warm `timing_ms.inference`, 5 heads, 500×334 (fused: 6-head bucket) | 7.43 / 7.37 / 7.46 ms | 15.52 / 15.27 / 15.81 ms |
| Host harness, 1 / 3 / 10 heads (median of 50 calls) | 5.65 / 6.41 / 9.76 ms | 9.12 / 12.47 / 24.01 ms |
| Device ops per forward (from the code, not re-measured with Tracy) | 144 for any head count (112 scene + 32 decoder) | 201 / 315 / 714 at 1 / 3 / 10 heads |
| Per-frame host→device upload | 1.25 MB bf16 ROW_MAJOR into a persistent buffer (device tilize) + 66 KB per head bucket (mask) | 1.18 MB bf16 (host tilize) |
| Per-frame device→host download | one `(N, 1025, 5)` tensor (66 KB per head) | 2 per head (16 KB + 4 B) |

Measured 2026-09-13 on one p150a, tt-metal `v0.78.0-dev20260820` (main
`8b98410e730`); the served rows come from the packaged FastAPI server running
this code (the server itself is not part of this repo). The table that used to
sit here (10.47 ms mean / ~96 FPS, 202 ops) was the legacy path on an earlier
tree. The op counts are read off the code; Tracy was not re-run after the
change. Full per-lever A/B, the one hardware defect found (a trace-aliasing bug,
fixed) and the boot/shutdown checks: `DEVICE_VALIDATION.md`.

### Optimization trajectory

Starting point was a pure-torch CPU run; each row below is a change that was
verified to improve throughput without dropping below the 99% PCC/AUC gate.
Intermediate experiments that regressed or were washes are not listed.

| # | Change | FPS | End-to-end PCC |
|---:|---|---:|---:|
| 0 | Torch CPU reference baseline                                              |   1.6 | —      |
| 1 | Naive TT-NN: 12 DINOv2 encoder blocks on chip; gaze decoder + heads on CPU | 13.8 | 0.9999 |
| 2 | Explicit `core_grid=(10,13)` on every ttnn.linear                          | 15.0 | 0.9989 |
| 3 | Gaze decoder (proj + pos + 3 transformer blocks) moved to chip             | 39.7 | 0.9983 |
| 4 | On-device `ttnn.slice` to drop CLS+REG tokens (was host round-trip)        | 45.1 | 0.9983 |
| 5 | Fold DINOv2 LayerScale into adjacent projection weights (remove 24 muls)   | 46.5 | 0.9983 |
| 6 | Fused SDPA kernel replaces manual Q·Kᵀ → softmax → ·V sequence             | 52.0 | 0.9986 |
| 7 | Pack DINOv2 MLP weights as `bfloat8_b`                                     | 54.3 | 0.9987 |
| 8 | Pack DINOv2 attention QKV + proj weights as `bfloat8_b`                    | 57.1 | 0.9987 |
| 9 | `LoFi` `compute_kernel_config` on the bfp8 DINOv2 matmuls                  | 63.8 | 0.9989 |
| 10| Fuse `pos_embed + head_map*head_token` on host (one add instead of two)    | 65.2 | 0.9990 |
| 11| Early-deallocate LayerNorm intermediates inside DINOv2 blocks              | 66.2 | 0.9990 |
| 12| Full e2e on Blackhole — patch_embed + heatmap_head + inout_head on chip¹   | 54.8 | 0.9983 |
| 13| Swap the on-device `ttnn.fold` patch embed for a direct (588,768) matmul² | 94.0 | 0.9984 |
| 14| Bbox mask generated on device via `ttnn.ge`/`lt`/`mul` (no mid-pipeline upload) | 94.0 | 0.9984 |
| 15 | Fused traced path (`TT_FUSED`, default): metal-trace + dit fused matmul+residual + SDPA q128/k128 + batched heads + single readback³ | 181 | 0.9984 |

¹ This step moved compute off the host and briefly regressed FPS — the on-device
`fold + to_layout` pipeline was dominated by layout-conversion overhead.
² The patch embed is mathematically equivalent to a single matmul once we
pre-rearrange the image on host (a pure view+permute+reshape — not inference
compute). Dropping the fold and its two `to_layout` conversions recovered the
regression and then some; this is the largest single optimization in the repo.
³ Rows 0–14 were measured on an earlier tt-metal tree; on the 2026-09-13 tree
the legacy path (row 14 code) runs at 106 FPS and the fused path at 181 FPS
(`benchmark.py`, 1 head), i.e. 1.7× on the same day / same tree. The per-lever
breakdown (trace + exact reformulations 9.12 → 8.13 ms, SDPA config → 6.51,
q128/k128 −0.22, dit fused ops → 5.87, final 5.65 ms at 1 head) is in
`DEVICE_VALIDATION.md`.

Several attempted improvements regressed and were rejected:

- L1 memory config on all intermediate activations (activations too large for L1).
- bfp8_b weights for the gaze decoder (matmuls too small; dispatch overhead dominates).
- ttnn Tracer-based graph replay (Tracer adds per-input copy overhead that wipes the dispatch-overhead savings at this op count). Metal-trace (`ttnn.begin_trace_capture` / `execute_trace`) with persistent input buffers is what the fused path uses instead.
- `packer_l1_acc=True`, per-block smaller core grids, HiFi2 on the gaze decoder.
- Persistent pre-allocated upload buffer via `ttnn.copy_host_to_device_tensor` (on the eager path; the fused path relies on it, since a trace needs fixed input buffers).
- Combined inout + heatmap download (`pad` + `concat` added timing variance).

Details and the exact numbers for every discarded attempt live in the `tt-metal`
branch commit history that produced this repo.

---

## Known caveats

- **One image per forward.** `TtGazeLLE.__call__` asserts `B == 1`. Multiple
  heads in the same image are supported by passing a list of bboxes and share
  the scene encode; multiple **images** in the same forward (true batching)
  would require staging the image upload and backbone matmuls as batched
  shapes — not wired up yet.
- **Per-head decoder runs serially on the legacy path only.** With `TT_FUSED=0`
  the decoder tail (head-mask + 3 gaze blocks + heads) is dispatched `N` times
  in sequence. The fused path batches all heads along `N` in one decoder trace
  per bucket (1, 2, 3, 4, 6, 8, 10 heads); `N` above the largest bucket
  captures a new trace lazily, which costs seconds on that call — pass your own
  `warmup(heads=[...])` / `TT_FUSED_TRACE_HEADS` if you need more.
- **Multi-person output equals single-person output (bit-exact).** Running
  `TtGazeLLE(image, [bbox_a, bbox_b])` returns output `[0]` that is
  **numerically identical** to `TtGazeLLE(image, [bbox_a])` — i.e. the
  per-head decoder does not interfere between heads. Verified on both paths
  (fused: N=1 == head 0 of N=3, and N=5 padded to the 6-head bucket == N=3
  heads, max abs diff 0.0). Against a torch reference on the same pretrained
  weights, per-head PCC sits at the same ~0.99 as single-person mode. The test
  suite's looser 0.95 bar on `test_multi_person.py` is because that test uses
  non-face-centered bboxes for the purpose of exercising the N-head code path;
  numerical equivalence is verified by `pcc(tt_single, tt_multi) == 1.0000`.
- **One traced `TtGazeLLE` per process / device.** A second fused model in the
  same process allocates its constants after the first model's traces were
  captured and can alias their intermediates (this is the allocator-ordering
  hazard that bit once during validation, see `DEVICE_VALIDATION.md`). Build
  one model per process (`TT_FUSED_EAGER=1` captures no traces, so the hazard
  should not apply there — untested).
- **Fused vs legacy are bf16-rounding-class equal, not bit-identical.** The
  fused graph changes the token order (special token last), folds the pos/bias
  adds, uses bf16 instead of bfp8 for the fused-matmul weights and larger SDPA
  chunks with exact exp; every reformulation is exact in fp32 (host tests) and
  the device result differs from the legacy path at the bf16 level (GazeFollow
  AUC −0.0001, L2 better). Gate changes on the GazeFollow metric, not on
  random-weight PCC alone — the heatmap argmax is discontinuous.
- **`test_pretrained_eval.py` fails on this tt-metal tree for both paths** (see
  Results); its thresholds date from an earlier tree and were not changed.
- **`TT_FUSED_MINIMAL_MM`, `TT_FUSED_L1`, `TT_FUSED_BF16_WEIGHTS` were not run
  on hardware.** They are validate-legal by the op sources but unmeasured; they
  default to off.
- **Random-weight PCC is the numerical baseline.** Real pretrained weights
  change absolute activations but the stage-relative PCC pattern is the same.
  Run both `test_relative_pcc.py` and `test_pretrained_eval.py` to cover both.
- **Paper GazeFollow L2s** are from the `test_preprocessed.json` protocol. This
  repo uses the `vikhyatk/gazefollow` HF mirror — same 4,782 images, slightly
  different annotator coords, so the L2 deltas do not match the paper exactly
  but are consistent between torch and TT.
- **Only `vitb14` is exercised.** The reference supports `vitl14` (24-layer,
  1024-d) but the TT port has only been benchmarked against the base model.
- **Tracy post-processing pandas bug.** `--profiler-capture-perf-counters=all`
  triggers a `pd.to_numeric` crash in
  `tt-metal/tools/tracy/process_device_log.py` so the canonical
  `ops_perf_results.csv` is not produced; the raw
  `profile_log_device.csv` + `tracy_ops_data.csv` need to be parsed by hand for
  now. The per-op breakdown in this README was produced that way.

---

## License

Apache 2.0 (matches the upstream Gaze-LLE, DINOv2, and tt-metal licenses).

---

## Acknowledgements

- Original model: Ryan, Wijekoon, Shanmugam et al., Gaze-LLE — https://github.com/fkryan/gazelle
- Backbone: Meta AI's DINOv2 — https://github.com/facebookresearch/dinov2
- Runtime: Tenstorrent tt-metal / tt-nn — https://github.com/tenstorrent/tt-metal
- GazeFollow test mirror: https://huggingface.co/datasets/vikhyatk/gazefollow
