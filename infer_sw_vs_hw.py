# -*- coding: utf-8 -*-
"""
infer_sw_vs_hw.py

用 HLS 的 weights.hpp 在軟體 structure 架構跑一次：
- preprocess 跟 notebook 一致：BGR->RGB、resize=640x320、INTER_AREA、0..255
- 載入 weights.hpp (conv1~7 offset 已在 champ_weight_loader 處理)
- forward -> (1,72,20,40)
- 轉成 (1,20,40,72) int32 out_buffer
- 用 notebook decode：postprocess_ultraspeed_nms_with_obj_th
- 畫框輸出圖片

Usage:
  python infer_sw_vs_hw.py --weights "C:\...\weights.hpp" --image "C:\...\00001.jpg" --out "vis.jpg"
"""

from __future__ import annotations
import argparse
import cv2
import numpy as np
import torch

import train
import structure
from champ_weight_loader import load_champ_weights_into_model


# ---- model config (match champion) ----
SHIFTS = [19, 15, 15, 15, 15, 15, 15, 15]
INC_BITS_NEW = [15, 13, 12, 11, 12, 12, 12, 12]
BIAS_BITS_NEW = [25, 21, 21, 21, 21, 21, 21, 21]
HEAD_BIAS_BITS_NEW = 11

IN_W, IN_H = 640, 320
NUM_CLASSES = 7

# ---- notebook decode helpers ----
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.clip(x, -20, 20)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)

def _nms_xyxy(boxes, scores, iou_th):
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(iou <= iou_th)[0] + 1]
    return keep

def postprocess_ultraspeed_nms_with_obj_th(
    out_buffer, orig_h, orig_w,
    in_h=320, in_w=640,
    num_classes=7,
    scale_bbox=256.0,
    scale_cls=512.0,
    scale_obj=512.0,
    obj_th=0.30,
    conf_th=0.07,
    iou_th=0.01,
    anchors=None,
    twth_clip=2.0,
    max_dets=200,
    agnostic_nms=False,
    return_score=True,
    verbose=False,
):
    # out_buffer: (1,20,40,72) OR (20,40,72)
    pred_int = out_buffer[0] if (isinstance(out_buffer, np.ndarray) and out_buffer.ndim == 4) else out_buffer
    if pred_int.shape != (20, 40, 72):
        raise ValueError(f"Expect (20,40,72), got {pred_int.shape}")

    if anchors is None:
        # anchors = np.array([[12, 10], [20, 14], [28, 18], [40, 26], [56, 36], [80, 52]], dtype=np.float32)
        anchors = np.array([[6, 7],[13, 10], [11, 20], [24, 16],[41, 30], [90, 64]], dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)

    attrs = 5 + num_classes  # 12
    A = 72 // attrs          # 6
    if anchors.shape != (A, 2):
        raise ValueError(f"anchors must be shape ({A},2), got {anchors.shape}")

    p = pred_int.astype(np.float32).reshape(20, 40, A, attrs)

    tx = p[..., 0] / scale_bbox
    ty = p[..., 1] / scale_bbox

    # NOTE: notebook 固定 tw/th 除以 512
    tw = p[..., 2] / 512.0
    th = p[..., 3] / 512.0

    tobj_raw = p[..., 4]
    tcls_raw = p[..., 5:]

    obj = _sigmoid(tobj_raw / scale_obj)
    cls_prob = _softmax(tcls_raw / scale_cls, axis=-1)

    cls_id = np.argmax(cls_prob, axis=-1)
    cls_p = np.max(cls_prob, axis=-1)

    score = obj * cls_p
    mask = (score > conf_th) & (obj > obj_th)
    if not np.any(mask):
        return []

    gy, gx = np.meshgrid(np.arange(20), np.arange(40), indexing="ij")
    gx = gx[..., None]
    gy = gy[..., None]
    stride_x = in_w / 40.0
    stride_y = in_h / 20.0

    cx = (_sigmoid(tx) + gx) * stride_x
    cy = (_sigmoid(ty) + gy) * stride_y

    tw = np.clip(tw, -twth_clip, twth_clip)
    th = np.clip(th, -twth_clip, twth_clip)

    aw = anchors[:, 0][None, None, :]
    ah = anchors[:, 1][None, None, :]
    bw = np.exp(tw) * aw
    bh = np.exp(th) * ah

    x1 = (cx - bw / 2.0)[mask]
    y1 = (cy - bh / 2.0)[mask]
    x2 = (cx + bw / 2.0)[mask]
    y2 = (cy + bh / 2.0)[mask]

    boxes_in = np.stack([x1, y1, x2, y2], axis=1)
    scores_1d = score[mask].reshape(-1)
    obj_1d = obj[mask].reshape(-1)
    cls_1d = cls_id[mask].reshape(-1)

    sx = orig_w / float(in_w)
    sy = orig_h / float(in_h)
    boxes = boxes_in.copy()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy

    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

    keep_global = []
    if agnostic_nms:
        keep_global = _nms_xyxy(boxes, scores_1d, iou_th)
    else:
        for c in np.unique(cls_1d):
            idx = np.where(cls_1d == c)[0]
            keep = _nms_xyxy(boxes[idx], scores_1d[idx], iou_th)
            keep_global.extend(idx[k] for k in keep)

    keep_global = np.array(keep_global, dtype=np.int32)
    if keep_global.size == 0:
        return []

    keep_global = keep_global[np.argsort(scores_1d[keep_global])[::-1]]
    keep_global = keep_global[:int(max_dets)]

    dets = []
    for i in keep_global:
        b = boxes[i]
        x = int(b[0])
        y = int(b[1])
        w = int(max(0, b[2] - b[0]))
        h = int(max(0, b[3] - b[1]))
        d = {"type": int(cls_1d[i]) + 1, "x": x, "y": y, "width": w, "height": h}
        if return_score:
            d["score"] = float(scores_1d[i])
            d["obj"] = float(obj_1d[i])
        dets.append(d)

    if verbose:
        print("cls_p min/max/mean:", float(cls_p.min()), float(cls_p.max()), float(cls_p.mean()))
        print("obj min/max/mean:", float(obj.min()), float(obj.max()), float(obj.mean()))
    return dets


# ---- preprocess exactly like notebook ----
def load_image_like_notebook(image_path: str):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    orig_h, orig_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().unsqueeze(0)  # 0..255
    return x, (orig_h, orig_w), rgb

def to_out_buffer_like_hw(y_pred: torch.Tensor) -> np.ndarray:
    # (1,72,20,40) -> (1,20,40,72) int32
    y = y_pred.detach().cpu().numpy()
    y = np.rint(y).astype(np.int32)
    y = np.transpose(y, (0, 2, 3, 1))
    return y

def draw_dets(image_path: str, dets, out_path: str, max_draw: int = 200):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    H, W = img.shape[:2]
    for d in dets[:max_draw]:
        x, y, w, h = int(d["x"]), int(d["y"]), int(d["width"]), int(d["height"])
        x1 = max(0, min(W - 1, x))
        y1 = max(0, min(H - 1, y))
        x2 = max(0, min(W - 1, x + max(0, w)))
        y2 = max(0, min(H - 1, y + max(0, h)))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        score = d.get("score", 0.0)
        cv2.putText(img, f"{d['type']}:{score:.3f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(out_path, img)
    print(f"[saved] {out_path} (boxes={len(dets)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="sw_vis.jpg")
    ap.add_argument("--device", default="cpu")

    # decode args (match your notebook run)
    ap.add_argument("--conf_th", type=float, default=0.07)
    ap.add_argument("--obj_th", type=float, default=0.30)
    ap.add_argument("--iou_th", type=float, default=0.01)
    ap.add_argument("--scale_bbox", type=float, default=256.0)
    ap.add_argument("--scale_cls", type=float, default=512.0)
    ap.add_argument("--scale_obj", type=float, default=512.0)
    ap.add_argument("--max_dets", type=int, default=200)
    ap.add_argument("--agnostic_nms", action="store_true")
    ap.add_argument("--verbose_decode", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1) build model
    model = structure.HLSLikeQATDetector(
        shifts=SHIFTS,
        inc_bits_new=INC_BITS_NEW,
        bias_bits_new=BIAS_BITS_NEW,
        head_bias_bits_new=HEAD_BIAS_BITS_NEW,
    ).to(device)
    model.eval()

    # 2) load weights.hpp (HLS-aligned)
    load_champ_weights_into_model(model, args.weights)

    # 3) preprocess like notebook
    x, (orig_h, orig_w), _ = load_image_like_notebook(args.image)
    x = x.to(device)

    # 4) forward
    with torch.no_grad():
        y_pred = model(x)   # (1,72,20,40)

    print("[y_pred] shape:", tuple(y_pred.shape))
    print("[y_pred] min/max:", float(y_pred.min()), float(y_pred.max()))

    out_buf = to_out_buffer_like_hw(y_pred)  # (1,20,40,72)
    

    # 5) decode (notebook)
    dets = postprocess_ultraspeed_nms_with_obj_th(
        out_buf, orig_h, orig_w,
        in_h=IN_H, in_w=IN_W,
        num_classes=NUM_CLASSES,
        scale_bbox=args.scale_bbox,
        scale_cls=args.scale_cls,
        scale_obj=args.scale_obj,
        obj_th=args.obj_th,
        conf_th=args.conf_th,
        iou_th=args.iou_th,
        max_dets=args.max_dets,
        agnostic_nms=args.agnostic_nms,
        return_score=True,
        verbose=args.verbose_decode,
    )

    print(f"[dets] count={len(dets)}")
    for i, d in enumerate(dets[:20]):
        print(f"  {i:02d} cls={d['type']-1} score={d['score']:.3f} obj={d['obj']:.3f} xywh=({d['x']},{d['y']},{d['width']},{d['height']})")

    # 6) draw
    draw_dets(args.image, dets, args.out)


if __name__ == "__main__":
    main()
