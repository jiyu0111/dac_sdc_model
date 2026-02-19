# -*- coding: utf-8 -*-
"""viz_predict_hpp.py

一次抽樣 N 張圖片，用指定的 weights.hpp 載入模型做推論，
把預測框畫回圖片並統一存到輸出資料夾。

假設你把這支檔案放在與 train.py / structure.py / champ_weight_loader.py 同一個目錄。

Example:
  python viz_predict_hpp.py --weights_hpp ./export_out/weights.hpp \
    --img_dir C:/Users/USER/Desktop/model/JPEGImages \
    --out_dir ./viz_out --num_images 50 --conf 0.25 --nms_iou 0.45

Optional:
  --label_dir 可額外把 GT 也畫上去(綠色)
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

import structure
import train
from champ_weight_loader import load_champ_weights_into_model


# 跟 train.py 裡一致
ANCHORS = [(6, 7), (13, 10), (11, 20), (24, 16), (41, 30), (90, 64)]
GRID_SIZE = (40, 20)  # (GW, GH)
IMG_SIZE = (640, 320)  # (W, H)

# 跟 train.main() 裡一致
SHIFTS = [19, 15, 15, 15, 15, 15, 15, 15]
INC_BITS_NEW = [15, 13, 12, 11, 12, 12, 12, 12]
BIAS_BITS_NEW = [25, 21, 21, 21, 21, 21, 21, 21]
HEAD_BIAS_BITS_NEW = 11


def list_images(img_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    ps = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    ps.sort()
    return ps


def make_class_colors(num_classes: int = 7, seed: int = 0) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    colors = []
    for _ in range(num_classes):
        # BGR colors for OpenCV
        colors.append((rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    return colors


@torch.no_grad()
def infer_one(
    model: torch.nn.Module,
    bgr: np.ndarray,
    device: str,
    conf_th: float,
    nms_iou: float,
    max_det: int,
    round_like_hw: bool,
):
    """Return pred dict from train.postprocess for a single image."""
    orig_h, orig_w = bgr.shape[:2]
    W, H = IMG_SIZE

    # resize to model input size
    bgr_rs = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr_rs, cv2.COLOR_BGR2RGB)

    x = (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .unsqueeze(0)
        .to(device)
    )

    y = model(x)  # (1,72,20,40)
    if round_like_hw:
        y = torch.round(y)

    pred = train.postprocess(
        y,
        ANCHORS,
        img_size=IMG_SIZE,
        grid_size=GRID_SIZE,
        conf_th=conf_th,
        iou_th=nms_iou,
        max_det=max_det,
    )[0]

    # scale boxes back to original image coords
    if pred["boxes"].numel() > 0:
        pb = pred["boxes"].numpy().copy()  # resized space (W,H)
        sx = orig_w / float(W)
        sy = orig_h / float(H)
        pb[:, [0, 2]] *= sx
        pb[:, [1, 3]] *= sy
        pred["boxes_orig"] = pb
    else:
        pred["boxes_orig"] = np.zeros((0, 4), dtype=np.float32)

    return pred


def draw_boxes(
    bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    colors: list[tuple[int, int, int]],
    thickness: int = 2,
):
    vis = bgr.copy()
    for (x1, y1, x2, y2), s, c in zip(boxes_xyxy, scores, classes):
        c = int(c)
        color = colors[c % len(colors)]
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), color, thickness)
        cv2.putText(
            vis,
            f"{c}:{float(s):.2f}",
            (x1i, max(0, y1i - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return vis


def try_load_gt(label_dir: Path, stem: str, orig_w: int, orig_h: int):
    """用 train.JsonDetDataset 的 JSON 格式讀 GT，並回傳原圖座標的 GT boxes/labels。

    注意：train.JsonDetDataset 會先 resize 再 scale；這裡我們直接在原圖座標讀。
    JSON 內的 x,y,width,height 是以原圖座標存的（從你的 dataset 寫法來看）。
    """
    import json

    lab_path = label_dir / f"{stem}.json"
    if not lab_path.exists():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

    TYPE_TO_CLASSID = train.TYPE_TO_CLASSID

    objs = json.load(open(lab_path, "r", encoding="utf-8"))
    boxes = []
    labels = []
    for o in objs:
        t = o.get("type", -1)
        if t not in TYPE_TO_CLASSID:
            continue
        x, y, w, h = o["x"], o["y"], o["width"], o["height"]
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            continue
        # clamp to image
        x1 = max(0.0, float(x))
        y1 = max(0.0, float(y))
        x2 = min(float(orig_w - 1), float(x + w))
        y2 = min(float(orig_h - 1), float(y + h))
        boxes.append([x1, y1, x2, y2])
        labels.append(TYPE_TO_CLASSID[t])

    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

    return np.array(boxes, np.float32), np.array(labels, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_hpp", type=str, required=True, help="path to export_out/*.hpp")
    ap.add_argument("--img_dir", type=str, required=True, help="image directory")
    ap.add_argument("--out_dir", type=str, default="viz_out", help="output directory")
    ap.add_argument("--num_images", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf", type=float, default=0.25, help="score threshold")
    ap.add_argument("--nms_iou", type=float, default=0.01)
    ap.add_argument("--max_det", type=int, default=3)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--round_like_hw", action="store_true", help="round head output before postprocess")
    ap.add_argument("--draw_gt", action="store_true", help="also draw GT boxes if --label_dir provided")
    ap.add_argument("--label_dir", type=str, default=None, help="label json directory (optional)")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # build model
    model = structure.HLSLikeQATDetector(
        shifts=SHIFTS,
        inc_bits_new=INC_BITS_NEW,
        bias_bits_new=BIAS_BITS_NEW,
        head_bias_bits_new=HEAD_BIAS_BITS_NEW,
    ).to(device)

    load_champ_weights_into_model(model, args.weights_hpp)
    model.eval()

    imgs = list_images(img_dir)
    if len(imgs) == 0:
        raise SystemExit(f"No images found in: {img_dir}")

    rng = random.Random(args.seed)
    pick = imgs if args.num_images >= len(imgs) else rng.sample(imgs, args.num_images)

    colors = make_class_colors(num_classes=7, seed=0)

    label_dir = Path(args.label_dir) if args.label_dir else None
    if args.draw_gt and (label_dir is None):
        print("[WARN] --draw_gt given but --label_dir is None, will skip GT.")

    print(f"device={device}  weights={args.weights_hpp}")
    print(f"images: total={len(imgs)}  selected={len(pick)}  out_dir={out_dir}")

    # write a simple index for convenience
    index_lines = []

    for p in tqdm(pick, desc="Visualizing"):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        orig_h, orig_w = bgr.shape[:2]
        stem = p.stem

        pred = infer_one(
            model,
            bgr,
            device=device,
            conf_th=args.conf,
            nms_iou=args.nms_iou,
            max_det=args.max_det,
            round_like_hw=args.round_like_hw,
        )

        vis = bgr.copy()

        # optional GT
        if args.draw_gt and label_dir is not None:
            gt_boxes, gt_labels = try_load_gt(label_dir, stem, orig_w, orig_h)
            for (x1, y1, x2, y2), c in zip(gt_boxes, gt_labels):
                color = (0, 255, 0)
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    vis,
                    f"GT{int(c)}",
                    (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        pb = pred["boxes_orig"]
        ps = pred["scores"].numpy() if isinstance(pred["scores"], torch.Tensor) else np.asarray(pred["scores"])
        pc = pred["classes"].numpy() if isinstance(pred["classes"], torch.Tensor) else np.asarray(pred["classes"])

        vis = draw_boxes(vis, pb, ps, pc, colors, thickness=2)

        out_path = out_dir / f"{stem}_pred.jpg"
        cv2.imwrite(str(out_path), vis)

        index_lines.append(f"{p.name}\t{out_path.name}\tndet={len(pb)}")

    (out_dir / "index.txt").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"[OK] saved {len(index_lines)} images to: {out_dir}")
    print(f"[OK] index: {out_dir / 'index.txt'}")


if __name__ == "__main__":
    main()
