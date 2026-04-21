# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generate media/source_{N}.png + media/target_{N}.png demo artifacts.

Pulls 4 clean images (single head bbox each, no pre-baked annotations) from the
GazeFollow test parquet, runs the TT-NN forward on a Blackhole p150a, and
writes the visualization — heatmap overlay, head bbox (green), predicted gaze
target (red ×), and a yellow arrow showing the gaze direction from bbox
center to the predicted target.

Run from repo root with::

    PYTHONPATH=$PWD:$TT_METAL_HOME:$TT_METAL_HOME/ttnn \
    TT_VISIBLE_DEVICES=<n> \
    python -m scripts.make_demo
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

# Pick 4 rows from the GazeFollow test parquet to get a mix of scenes.
_SAMPLE_INDICES = [0, 800, 2300, 4000]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PARQUET = Path(os.environ.get("TT_GAZE_LLE_DATA", _REPO_ROOT / "data")) / "gazefollow" / "test.parquet"
_MEDIA = _REPO_ROOT / "media"


def visualize(img_pil: Image.Image, bbox, heatmap_64: torch.Tensor, inout_score: float, out_path: Path) -> None:
    W, H = img_pil.size
    hm = heatmap_64.float().cpu().numpy()
    hm_pil = Image.fromarray((hm * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    hm_arr = np.asarray(hm_pil) / 255.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 10 * H / W))
    ax.imshow(img_pil)
    ax.imshow(hm_arr, cmap="jet", alpha=0.45)

    xmin, ymin, xmax, ymax = bbox
    ax.add_patch(plt.Rectangle(
        (xmin * W, ymin * H), (xmax - xmin) * W, (ymax - ymin) * H,
        fill=False, edgecolor="lime", linewidth=3,
    ))

    py, px = np.unravel_index(hm_arr.argmax(), hm_arr.shape)
    ax.scatter([px], [py], c="red", s=220, marker="x", linewidths=4)

    cx = (xmin + xmax) / 2 * W
    cy = (ymin + ymax) / 2 * H
    ax.annotate(
        "", xy=(px, py), xytext=(cx, cy),
        arrowprops=dict(arrowstyle="->", color="yellow", lw=4,
                         shrinkA=8, shrinkB=8, mutation_scale=22),
    )

    ax.set_title(f"gaze target (red ×), gaze direction (yellow arrow),  inout={inout_score:.3f}", fontsize=13)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def _load_samples_from_parquet(path: Path, indices):
    df = pd.read_parquet(path)
    out = []
    for idx in indices:
        row = df.iloc[idx]
        pil = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        hb = row["gazes"][0]["head_bbox"]
        bbox = (float(hb["xmin"]), float(hb["ymin"]), float(hb["xmax"]), float(hb["ymax"]))
        out.append((pil, bbox))
    return out


def main() -> None:
    import ttnn

    from gaze_lle.reference.load_pretrained import load_pretrained
    from gaze_lle.reference.torch_gaze_lle import build_gaze_lle
    from gaze_lle.tt.tt_gaze_lle import TtGazeLLE

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    ref = build_gaze_lle("vitb14", inout=True).eval()
    load_pretrained(ref, verbose=False)

    _MEDIA.mkdir(parents=True, exist_ok=True)
    samples = _load_samples_from_parquet(_PARQUET, _SAMPLE_INDICES)

    device_id = int(os.environ.get("GAZE_LLE_DEVICE", "0"))
    d = ttnn.open_device(device_id=device_id)
    try:
        tt_model = TtGazeLLE(ref, d, inout=True)

        tf = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Warm-up so the first captured forward is steady-state.
        _ = tt_model(tf(samples[0][0]).unsqueeze(0), [samples[0][1]])

        for i, (img_pil, bbox) in enumerate(samples, start=1):
            src_path = _MEDIA / f"source_{i}.png"
            img_pil.save(src_path)

            out = tt_model(tf(img_pil).unsqueeze(0), [bbox])
            heatmap = out["heatmap"][0]
            inout_score = float(out["inout"][0]) if out["inout"] is not None else 0.0

            tgt_path = _MEDIA / f"target_{i}.png"
            visualize(img_pil, bbox, heatmap, inout_score, tgt_path)
            print(f"  {src_path.name}  ({img_pil.size[0]}x{img_pil.size[1]})  bbox={bbox}  inout={inout_score:.3f}  →  {tgt_path.name}")
    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
