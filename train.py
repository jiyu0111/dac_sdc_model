import os
import json
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.ops import box_iou  # optional (you also have your own IoU below)
from tqdm.auto import tqdm
from champ_weight_loader import load_champ_weights_into_model
from pathlib import Path

import structure
import export
import evaluate


# =============================
# Head output fixed-point scales
# - SCALE_XY: tx/ty (sigmoid inputs for center offsets)
# - SCALE_WH: tw/th (exp inputs for width/height)
# - SCALE_OC: objectness + class logits (sigmoid inputs)
# IMPORTANT: these must match the right-shifts / LUT input Q-format used in your HLS postprocess.
# =============================
SCALE_XY = 256.0
SCALE_WH = 512.0
SCALE_OC = 512.0

# Backward-compat alias used in older code paths
LOGIT_SCALE = SCALE_OC


# =============================
# Data
# =============================
TYPE_TO_CLASSID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}


class JsonDetDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=(640, 320)):  # (W,H)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.W, self.H = img_size
        self.ids = sorted(
            [
                os.path.splitext(f)[0]
                for f in os.listdir(label_dir)
                if f.endswith(".json")
            ]
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        stem = self.ids[idx]
        img_path = os.path.join(self.img_dir, stem + ".jpg")
        lab_path = os.path.join(self.label_dir, stem + ".json")

        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(img_path)

        orig_h, orig_w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (self.W, self.H), interpolation=cv2.INTER_AREA)


        sx = self.W / orig_w
        sy = self.H / orig_h

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

            # scale to resized image
            x *= sx
            y *= sy
            w *= sx
            h *= sy

            # xyxy
            boxes.append([x, y, x + w, y + h])
            labels.append(TYPE_TO_CLASSID[t])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)


        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        img = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .contiguous()
            .float()
        )
        return img, boxes, labels, stem


def collate_fn(batch):
    imgs, boxes, labels, stems = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, boxes, labels, stems


# =============================
# YOLO target / utils
# =============================
def bbox_iou_wh(wh1, wh2):
    """
    wh1: (N,2), wh2: (A,2) in pixels
    return: (N,A)
    """
    w1, h1 = wh1[:, 0:1], wh1[:, 1:2]
    w2, h2 = wh2[:, 0], wh2[:, 1]
    inter = torch.min(w1, w2) * torch.min(h1, h2)
    union = (w1 * h1) + (w2 * h2) - inter
    return inter / (union + 1e-9)


def box_iou_xyxy(boxes1, boxes2):
    """
    boxes1: (N,4) xyxy
    boxes2: (M,4) xyxy
    return: (N,M)
    """
    a1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    a2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    union = a1[:, None] + a2[None, :] - inter + 1e-9
    return inter / union


def nms_xyxy(boxes, scores, iou_th=0.45, topk=300):
    """
    boxes: (N,4)
    scores: (N,)
    return: keep indices
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    scores, idx = scores.sort(descending=True)
    idx = idx[:topk]

    keep = []
    while idx.numel() > 0:
        i = idx[0]
        keep.append(i)
        if idx.numel() == 1:
            break

        ious = box_iou_xyxy(boxes[i].unsqueeze(0), boxes[idx[1:]]).squeeze(0)
        idx = idx[1:][ious <= iou_th]

    return torch.stack(keep)


def build_yolo_targets_v2(
    boxes_list,
    labels_list,
    anchors,
    img_size=(640, 320),
    grid_size=(40, 20),
    twth_clip: float = 2.0,
    device="cuda",
):
    W, H = img_size
    GW, GH = grid_size
    A = len(anchors)

    anchors_t = torch.tensor(anchors, dtype=torch.float32, device=device)  # (A,2)

    obj_t = torch.zeros((len(boxes_list), A, GH, GW), device=device)
    tx_t = torch.zeros_like(obj_t)
    ty_t = torch.zeros_like(obj_t)
    tw_t = torch.zeros_like(obj_t)
    th_t = torch.zeros_like(obj_t)
    tw_t = tw_t
    th_t = th_t

    cls_id_t = torch.full(
        (len(boxes_list), A, GH, GW),
        -1,
        device=device,
        dtype=torch.long,
    )
    gt_idx_t = torch.full(
        (len(boxes_list), A, GH, GW),
        -1,
        device=device,
        dtype=torch.long,
    )

    for b, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
        if len(boxes) == 0:
            continue

        boxes = boxes.to(device)
        labels = labels.to(device)

        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
        bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)

        gx = cx / W * GW
        gy = cy / H * GH

        gi = gx.long().clamp(0, GW - 1)
        gj = gy.long().clamp(0, GH - 1)

        ious = bbox_iou_wh(torch.stack([bw, bh], dim=1), anchors_t)  # (N,A)
        best_a = torch.argmax(ious, dim=1)

        for n in range(boxes.shape[0]):
            a = best_a[n].item()
            i = gi[n].item()
            j = gj[n].item()

            obj_t[b, a, j, i] = 1.0
            tx_t[b, a, j, i] = gx[n] - i
            ty_t[b, a, j, i] = gy[n] - j
            tw = torch.log(bw[n] / anchors_t[a, 0] + 1e-9).clamp(-twth_clip, twth_clip)
            th = torch.log(bh[n] / anchors_t[a, 1] + 1e-9).clamp(-twth_clip, twth_clip)
            tw_t[b,a,j,i] = tw
            th_t[b,a,j,i] = th
            cls_id_t[b, a, j, i] = labels[n].item()
            gt_idx_t[b, a, j, i] = n

    return obj_t, tx_t, ty_t, tw_t, th_t, cls_id_t, gt_idx_t


def iou_one_to_one(b1, b2, eps=1e-9):
    """
    b1,b2: (K,4) xyxy
    """
    ix1 = torch.max(b1[:, 0], b2[:, 0])
    iy1 = torch.max(b1[:, 1], b2[:, 1])
    ix2 = torch.min(b1[:, 2], b2[:, 2])
    iy2 = torch.min(b1[:, 3], b2[:, 3])

    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    a1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
    return inter / (a1 + a2 - inter + eps)


# =============================
# Loss
# =============================
def yolo_loss_v2(
    y_pred,
    targets,
    boxes_list,
    anchors,
    img_size=(640, 320),
    grid_size=(40, 20),
    num_classes=7,
    ignore_iou=0.5,
    epoch=0, obj_iou_start=150
):
    B, _, GH, GW = y_pred.shape
    A = len(anchors)
    C = num_classes

    y_raw = (
        y_pred.view(B, A, 5 + C, GH, GW)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )

    # fixed-point -> float logits
    tx = y_raw[..., 0] / SCALE_XY
    ty = y_raw[..., 1] / SCALE_XY
    tw = y_raw[..., 2] / SCALE_WH
    th = y_raw[..., 3] / SCALE_WH
    obj_logit = y_raw[..., 4] / SCALE_OC
    cls_logits = y_raw[..., 5:] / SCALE_OC

    obj_t, tx_t, ty_t, tw_t, th_t, cls_id_t, gt_idx_t = targets
    pos = obj_t > 0.5

    
    pred_boxes, pred_obj, _ = decode_yolo_head(
        y_pred,
        anchors,
        img_size,
        grid_size,
    )
    pred_boxes = pred_boxes.view(B, A, GH, GW, 4)
    
    obj_t_soft = obj_t.clone()

    if (epoch >= obj_iou_start):
        for b in range(B):
            if len(boxes_list[b]) == 0:
                continue
            gt = boxes_list[b].to(y_pred.device)  # (Ng,4)

            pos_b = pos[b]  # (A,GH,GW)
            if not pos_b.any():
                continue

           
            gt_idx = gt_idx_t[b][pos_b]  
            valid = gt_idx >= 0
            if not valid.any():
                continue

            pb = pred_boxes[b][pos_b][valid]          # (K,4)
            gb = gt[gt_idx[valid]]                    # (K,4)
            iou = iou_one_to_one(pb, gb).detach().clamp(0, 1)  # (K,)

            
            tmp = obj_t_soft[b][pos_b]
            tmp[valid] = 0.5 + 0.5 * iou
            obj_t_soft[b][pos_b] = tmp 


    ignore = torch.zeros((B, A, GH, GW), dtype=torch.bool, device=y_pred.device)
    for b in range(B):
        if len(boxes_list[b]) == 0:
            continue
        gt = boxes_list[b].to(y_pred.device)
        pb = pred_boxes[b].reshape(-1, 4)  # (A*GH*GW,4)
        iou = box_iou_xyxy(pb, gt).max(dim=1).values
        ignore[b] = (iou.view(A, GH, GW) > ignore_iou)

    ignore = ignore & (~pos)
    neg = (~pos) & (~ignore)

    # ---------- obj: count-aware ----------
    bce_obj = F.binary_cross_entropy_with_logits(obj_logit, obj_t_soft, reduction="none")
    loss_obj_pos = bce_obj[pos].mean() if pos.any() else (obj_logit.sum() * 0.0)
    loss_obj_neg = bce_obj[neg]
    if loss_obj_neg.numel():
        k = min(loss_obj_neg.numel(), max(256, int(pos.sum().item()) * 10))
        loss_obj_neg = loss_obj_neg.topk(k).values.mean()
    else:
        loss_obj_neg = obj_logit.sum() * 0.0
    # loss_obj_neg = loss_obj_neg.mean()

    lambda_noobj = 2.5
    loss_obj = loss_obj_pos + lambda_noobj * loss_obj_neg

    # ---------- box ----------
    if pos.any():
        loss_xy = (
            F.binary_cross_entropy_with_logits(tx[pos], tx_t[pos]) +
            F.binary_cross_entropy_with_logits(ty[pos], ty_t[pos])
        )


        # wh: log-space regression
        loss_wh = (
            F.mse_loss(tw[pos], tw_t[pos], reduction="mean")
            + F.mse_loss(th[pos], th_t[pos], reduction="mean")
        )

        cls_logits_pos = cls_logits[pos]
        cls_target_pos = cls_id_t[pos]
        K = cls_logits_pos.shape[0]
        t = torch.zeros((K, num_classes), device=cls_logits_pos.device)
        t[torch.arange(K), cls_target_pos] = 1.0
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits_pos, t, reduction="mean")
    else:
        loss_xy = obj_logit.sum() * 0.0
        loss_wh = obj_logit.sum() * 0.0
        loss_cls = obj_logit.sum() * 0.0

    lambda_coord = 20
    return lambda_coord * (loss_xy + loss_wh) +  1.0 * loss_obj + 1.0 * loss_cls



@torch.no_grad()
def decode_yolo_head(
    y_pred,
    anchors,
    img_size=(640, 320),
    grid_size=(40, 20),
    num_classes=7,
    scale_bbox: float = 256.0,   # tx/ty
    scale_twth: float = 512.0,   # tw/th (NOTE: notebook fixed 512)
    scale_obj: float = 512.0,    # obj
    scale_cls: float = 512.0,    # cls
    twth_clip: float = 2.0,
):
    """
    Match your numpy ultraspeed decode:
      tx,ty = raw/scale_bbox
      tw,th = raw/512.0 (fixed)
      obj   = sigmoid(raw/scale_obj)
      cls   = sigmoid(raw/scale_cls)

    Returns:
      boxes: (B, A*GH*GW, 4)  in resized-image xyxy (pixel)
      obj:   (B, A*GH*GW)     sigmoid(obj_logit)
      cls_logits: (B, A*GH*GW, C)  logits already scaled by scale_cls
    """
    B, _, GH, GW = y_pred.shape
    A = len(anchors)
    C = num_classes

    anchors_t = torch.tensor(anchors, dtype=y_pred.dtype, device=y_pred.device).view(1, A, 1, 1, 2)

    # (B, A, GH, GW, 5+C)
    y_raw = (
        y_pred.view(B, A, 5 + C, GH, GW)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )

    tx = y_raw[..., 0] / float(scale_bbox)
    ty = y_raw[..., 1] / float(scale_bbox)

    # NOTE: ultraspeed notebook fixed tw/th divisor to 512
    tw = y_raw[..., 2] / float(scale_twth)
    th = y_raw[..., 3] / float(scale_twth)

    obj_logit = y_raw[..., 4] / float(scale_obj)
    cls_logits = y_raw[..., 5:] / float(scale_cls)

    gy = torch.arange(GH, device=y_pred.device).view(1, 1, GH, 1)
    gx = torch.arange(GW, device=y_pred.device).view(1, 1, 1, GW)

    stride_x = img_size[0] / float(GW)
    stride_y = img_size[1] / float(GH)

    cx = (torch.sigmoid(tx) + gx) * stride_x
    cy = (torch.sigmoid(ty) + gy) * stride_y

    tw = tw.clamp(-twth_clip, twth_clip)
    th = th.clamp(-twth_clip, twth_clip)

    bw = torch.exp(tw) * anchors_t[..., 0]
    bh = torch.exp(th) * anchors_t[..., 1]

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(B, -1, 4)
    obj = torch.sigmoid(obj_logit).view(B, -1)
    cls_logits = cls_logits.view(B, -1, C)

    return boxes, obj, cls_logits


@torch.no_grad()
def postprocess(
    y_pred,
    anchors,
    img_size=(640, 320),
    grid_size=(40, 20),
    conf_th=0.20,
    obj_th=0.3,
    iou_th=0.01,
    max_det = 5,
    agnostic_nms=False,
):
    """
    Match ultraspeed notebook:
      score = obj * max(sigmoid(cls))
      mask  = (score>conf_th) & (obj>obj_th)
      NMS: class-wise unless agnostic_nms=True
      keep top max_det by score
    Return: list[B] of dict {boxes,scores,classes} (0-based classes)
    """
    boxes, obj, cls_logits = decode_yolo_head(
        y_pred,
        anchors,
        img_size=img_size,
        grid_size=grid_size,
        num_classes=7,
        scale_bbox=SCALE_XY,   # 256
        scale_twth=512.0,      # fixed
        scale_obj=SCALE_OC,    # 512
        scale_cls=SCALE_OC,    # 512
        twth_clip=2.0,
    )
    B, N, C = cls_logits.shape

    cls_prob = F.sigmoid(cls_logits)      # (B,N,C)
    cls_score, cls_id = cls_prob.max(dim=-1)      # (B,N)
    score = obj * cls_score                       # (B,N)

    results = []
    for b in range(B):
        m = (score[b] > conf_th) 
        # & (obj[b] > obj_th)
        if m.sum() == 0:
            results.append(
                {
                    "boxes": torch.empty((0, 4)),
                    "scores": torch.empty((0,)),
                    "classes": torch.empty((0,), dtype=torch.long),
                }
            )
            continue

        bb = boxes[b][m]
        ss = score[b][m]
        cc = cls_id[b][m]

        if agnostic_nms:
            keep = nms_xyxy(bb, ss, iou_th=iou_th, topk=3000)
        else:
            keep_all = []
            for k in range(C):
                mk = (cc == k)
                if mk.sum() == 0:
                    continue
                kk = nms_xyxy(bb[mk], ss[mk], iou_th=iou_th, topk=3000)
                idx_in_bb = torch.nonzero(mk, as_tuple=False).squeeze(1)[kk]
                keep_all.append(idx_in_bb)
            keep = torch.cat(keep_all, dim=0) if len(keep_all) else torch.empty((0,), dtype=torch.long, device=bb.device)

        if keep.numel() == 0:
            results.append(
                {
                    "boxes": torch.empty((0, 4)),
                    "scores": torch.empty((0,)),
                    "classes": torch.empty((0,), dtype=torch.long),
                }
            )
            continue

        # top max_det by score
        ss2 = ss[keep]
        top = ss2.argsort(descending=True)[:max_det]
        keep = keep[top]

        results.append(
            {
                "boxes": bb[keep].cpu(),
                "scores": ss[keep].cpu(),
                "classes": cc[keep].cpu(),
            }
        )

    return results

# =============================
# Debug
# =============================
@torch.no_grad()
def max_iou_all_boxes(y_pred, boxes_gt, anchors):
    boxes_all, obj_all, cls_logits_all = decode_yolo_head(
        y_pred,
        anchors,
        img_size=(640, 320),
        grid_size=(40, 20),
        num_classes=7,
    )
    pb = boxes_all[0]
    if boxes_gt.numel() == 0:
        return 0.0
    iou = box_iou_xyxy(pb, boxes_gt.to(pb.device)).max(dim=1).values
    return float(iou.max().item())


@torch.no_grad()
def dump_debug(
    model,
    ds_all,
    device,
    anchors,
    sample_index=0,
    img_size=(640, 320),
    grid_size=(40, 20),
    num_classes=7,
    save_path="debug_pred.jpg",
    conf_th=0,
    obj_th=0.30,
    iou_th=0.01,
    max_det=5,
    agnostic_nms=False,
    round_like_hw=True,
    save_path_orig=True,  
):
    model.eval()

    # ---------- 1) sample ----------
    img, boxes_gt, labels_gt, stem = ds_all[sample_index]
    W, H = img_size

    # ---------- 2) forward ----------
    x = img.unsqueeze(0).to(device, non_blocking=True)
    y = model(x)

    B, _, GH, GW = y.shape
    A = len(anchors)

    
    boxes_all, obj_all, cls_logits_all = decode_yolo_head(
        y,
        anchors,
        img_size=img_size,
        grid_size=grid_size,
        num_classes=num_classes,
    )
    pb_all = boxes_all[0]
    gt_dev = boxes_gt.to(pb_all.device)

    if gt_dev.numel() == 0:
        all_iou = 0.0
        ious_per_pred = None
    else:
        ious_per_pred = box_iou_xyxy(pb_all, gt_dev).max(dim=1).values
        all_iou = float(ious_per_pred.max().item())

    print("all-box max IoU:", all_iou)

    # ---------- 4) obj map stats ----------
    yy = (
        y.view(B, A, 5 + num_classes, GH, GW)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )
    obj_map = torch.sigmoid(yy[0, ..., 4] / float(SCALE_OC))

    # ---------- 5) bestIoU pred ----------
    if ious_per_pred is not None:
        k = int(ious_per_pred.argmax().item())
        best_iou = float(ious_per_pred[k].item())
        obj_k = float(obj_all[0, k].item())
        cls_prob = torch.sigmoid(cls_logits_all[0, k])
        cls_id = int(cls_prob.argmax().item())
        cls_p = float(cls_prob.max().item())
        score = obj_k * cls_p
        print(
            f"[bestIoU box] k={k} IoU={best_iou:.3f} "
            f"obj={obj_k:.3f} cls={cls_id}:{cls_p:.3f} score={score:.3f}"
        )

    # ---------- 6) GT positive cells ----------
    obj_t, tx_t, ty_t, tw_t, th_t, cls_id_t, _ = build_yolo_targets_v2(
        [boxes_gt],
        [labels_gt],
        anchors=anchors,
        img_size=img_size,
        grid_size=grid_size,
        device=device,
    )
    pos_idx = torch.nonzero(obj_t[0] > 0.5, as_tuple=False)
    print("GT positives count =", int(pos_idx.size(0)))

    if pos_idx.numel() > 0:
        pos_vals = obj_map[pos_idx[:, 0], pos_idx[:, 1], pos_idx[:, 2]]
        print(
            "obj on positive cells: mean/max =",
            float(pos_vals.mean().item()),
            float(pos_vals.max().item()),
        )
        print(
            "obj on positive cells (first 10):",
            [float(v) for v in pos_vals[:10].detach().cpu().tolist()],
        )

    pb_grid = boxes_all.view(1, A, GH, GW, 4)[0]
    for t in range(min(10, pos_idx.size(0))):
        aa, jj, ii = [int(v) for v in pos_idx[t].tolist()]

        t_tx = float(tx_t[0, aa, jj, ii].item())
        t_ty = float(ty_t[0, aa, jj, ii].item())
        t_tw = float(tw_t[0, aa, jj, ii].item())
        t_th = float(th_t[0, aa, jj, ii].item())
        t_cls = int(cls_id_t[0, aa, jj, ii].item())

        p_tx = float(torch.sigmoid(yy[0, aa, jj, ii, 0] / float(SCALE_XY)).item())
        p_ty = float(torch.sigmoid(yy[0, aa, jj, ii, 1] / float(SCALE_XY)).item())
        p_tw = float((yy[0, aa, jj, ii, 2] / float(SCALE_WH)).item())
        p_th = float((yy[0, aa, jj, ii, 3] / float(SCALE_WH)).item())
        p_obj = float(torch.sigmoid(yy[0, aa, jj, ii, 4] / float(SCALE_OC)).item())

        p_cls_prob = torch.sigmoid(yy[0, aa, jj, ii, 5:] / float(SCALE_OC))
        p_cls = int(p_cls_prob.argmax().item())
        p_cls_p = float(p_cls_prob.max().item())

        p_box = pb_grid[aa, jj, ii].unsqueeze(0)
        iou_here = 0.0
        if gt_dev.numel() > 0:
            iou_here = float(box_iou_xyxy(p_box, gt_dev).max().item())

        print(
            f" [pos{t}] (a,j,i)=({aa},{jj},{ii}) IoU={iou_here:.3f} "
            f"obj={p_obj:.3f} cls={p_cls}:{p_cls_p:.3f} | "
            f"tx={p_tx:.3f}/{t_tx:.3f} ty={p_ty:.3f}/{t_ty:.3f} "
            f"tw={p_tw:.3f}/{t_tw:.3f} th={p_th:.3f}/{t_th:.3f} "
            f"tcls={t_cls}"
        )

    # ---------- 7) postprocess ----------
    y_post = torch.round(y) if round_like_hw else y
    pred = postprocess(
        y_post,
        anchors,
        img_size=img_size,
        grid_size=grid_size,
        conf_th=conf_th,
        obj_th=obj_th,
        iou_th=iou_th,
        max_det=max_det,
        agnostic_nms=agnostic_nms,
    )[0]

    pb20 = pred["boxes"]
    ps20 = pred["scores"]
    pc20 = pred["classes"]

    rgb = (img.permute(1, 2, 0).cpu().numpy()).clip(0, 255).astype(np.uint8)
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # draw GT (green)
    for (x1, y1, x2, y2) in boxes_gt.numpy():
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # draw pred (red)
    if pb20.numel():
        for (x1, y1, x2, y2), s, c in zip(pb20.numpy(), ps20.numpy(), pc20.numpy()):
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(
                vis,
                f"{int(c)}:{float(s):.2f}",
                (int(x1), max(0, int(y1) - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

    cv2.imwrite(save_path, vis)
    print(f"saved {save_path} stem={stem} gt={len(boxes_gt)} pred={int(pb20.shape[0])}")

    # --- 可選：再存一張畫在「原圖」上的（跟 infer 對齊） ---
    if save_path_orig is not None:
        orig_path = os.path.join(ds_all.img_dir, stem + ".jpg")
        orig_bgr = cv2.imread(orig_path)
        if orig_bgr is not None:
            oh, ow = orig_bgr.shape[:2]
            sx = ow / float(W)
            sy = oh / float(H)

            vis2 = orig_bgr.copy()

            # GT -> scale 回原圖
            if boxes_gt.numel():
                gt2 = boxes_gt.numpy().copy()
                gt2[:, [0, 2]] *= sx
                gt2[:, [1, 3]] *= sy
                for (x1, y1, x2, y2) in gt2:
                    cv2.rectangle(vis2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # pred -> scale 回原圖
            if pb20.numel():
                pb2 = pb20.numpy().copy()
                pb2[:, [0, 2]] *= sx
                pb2[:, [1, 3]] *= sy
                for (x1, y1, x2, y2), s, c in zip(pb2, ps20.numpy(), pc20.numpy()):
                    cv2.rectangle(vis2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(
                        vis2,
                        f"{int(c)}:{float(s):.2f}",
                        (int(x1), max(0, int(y1) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

            cv2.imwrite(save_path_orig, vis2)
            print(f"saved {save_path_orig} (orig) stem={stem}")

    # ---------- extra stats (保留原本) ----------
    cls_score_all = torch.sigmoid(cls_logits_all[0]).max(dim=-1).values
    score_all = obj_all[0] * cls_score_all
    print("score mean/std/max:", float(score_all.mean()), float(score_all.std()), float(score_all.max()))

    topk = score_all.topk(20).indices
    if boxes_gt.numel():
        iou_topk = box_iou_xyxy(boxes_all[0][topk].cpu(), boxes_gt).max().item()
    else:
        iou_topk = 0.0
    print("top20-by-score max IoU:", iou_topk)

    if ious_per_pred is not None:
        k = int(ious_per_pred.argmax().item())
        rank = int((score_all > score_all[k]).sum().item()) + 1
        print("bestIoU score rank:", rank)
    else:
        print("bestIoU score rank: (no GT)")


# =============================
# Main
# =============================

ANCHORS = [(6, 7), (13, 10), (11, 20), (24, 16), (41, 30), (90, 64)]

def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    SHIFTS = [19, 15, 15, 15, 15, 15, 15, 15]
    INC_BITS_NEW = [15, 13, 12, 11, 12, 12, 12, 12]
    BIAS_BITS_NEW = [25, 21, 21, 21, 21, 21, 21, 21]
    HEAD_BIAS_BITS_NEW = 11

    # dataset / dataloader 先建好（等下要做 bnq calibration 用）
    ds_all = JsonDetDataset(
        img_dir=r"image path", #data could be found at https://drive.google.com/file/d/1ceQ5y_rCReSZ26HzzCf2muDNbovjyl5k/view?usp=share_link
        label_dir=r"label path",#data could be found at https://drive.google.com/file/d/1ceQ5y_rCReSZ26HzzCf2muDNbovjyl5k/view?usp=share_link
        img_size=(640, 320), 
    )

    indices = list(range(len(ds_all)))
    random.Random(42).shuffle(indices)
    val_ratio = 0.1
    val_n = int(len(indices) * val_ratio)
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    ds_train = Subset(ds_all, train_idx)
    ds_val = Subset(ds_all, val_idx)

    dl_train = DataLoader(
        ds_train,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ---- model ----
    model = structure.HLSLikeQATDetector(
        shifts=SHIFTS,
        inc_bits_new=INC_BITS_NEW,
        bias_bits_new=BIAS_BITS_NEW,
        head_bias_bits_new=HEAD_BIAS_BITS_NEW,
    ).to(device)


    # INIT_HPP = r"\weights.hpp"  
    BASE_DIR = Path(__file__).resolve().parent
    INIT_HPP = str(BASE_DIR / "initial_weight.h")
    USE_CHAMP_INIT = True
    DO_BNQ_CALIB = False   

    if USE_CHAMP_INIT:
        load_champ_weights_into_model(model, INIT_HPP)
        print(f"[OK] loaded champ weights: {INIT_HPP}")
    else:
        # scratch 才做這個
        with torch.no_grad():
            for _, m in model.named_modules():
                if isinstance(m, structure.QuantConv2dInt4):
                    m.conv.weight.uniform_(-2, 2)
                    if m.conv.bias is not None:
                        m.conv.bias.zero_()



    # -------------------------
    # 兩階段訓練：先 head-only，再全網路微調
    # -------------------------
    def set_head_only(m: torch.nn.Module, head_only: bool):
        for p in m.parameters():
            p.requires_grad = (not head_only)
        # 只開 head
        for p in m.head.parameters():
            p.requires_grad = True

    WARMUP_EPOCHS = 200

    # Phase1: head-only (lr 大一點)
    set_head_only(model, True)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    dump_debug(
        model, ds_all, device, ANCHORS,
        sample_index=0,
        save_path="debug_pred_resize.jpg",
        save_path_orig="debug_pred_orig.jpg",  
        conf_th=0, obj_th=0.30, iou_th=0.01, max_det=5,
        agnostic_nms=False,
        round_like_hw=True,                   
    )

    # -------------------------
    # train loop
    # -------------------------
    EPOCHS = 251
    for epoch in range(EPOCHS):
        model.train()
        if epoch == WARMUP_EPOCHS:
            set_head_only(model, False)
            opt = torch.optim.Adam(model.parameters(), lr=5e-5)
            print("[stage] unfreeze all, lr=1e-4")

        running = 0.0
        nstep = 0

        pbar = tqdm(dl_train, desc=f"Train epoch {epoch}", leave=False)
        for imgs, boxes_list, labels_list, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)

            y_pred = model(imgs)

            targets = build_yolo_targets_v2(
                boxes_list, labels_list,
                anchors=ANCHORS,
                img_size=(640, 320),
                grid_size=(40, 20),
                device=device,
            )

            loss = yolo_loss_v2(
                y_pred,
                targets,
                boxes_list=boxes_list,
                anchors=ANCHORS,
                img_size=(640, 320),
                grid_size=(40, 20),
                num_classes=7,
                ignore_iou=0.5,
                epoch = epoch
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            nstep += 1
            pbar.set_postfix(loss=float(loss.item()), avg=running / max(nstep, 1))

        train_loss = running / max(nstep, 1)
        print(f"epoch {epoch:03d} | train_loss={train_loss:.4f}")

        dump_debug(
            model, ds_all, device, ANCHORS,
            sample_index=0,
            save_path="debug_pred_resize.jpg",
            save_path_orig="debug_pred_orig.jpg",   
            conf_th=0, obj_th=0.30, iou_th=0.01, max_det=5,
            agnostic_nms=False,
            round_like_hw=True,                     
        )

        if (epoch % 10) == 0:
            metrics = evaluate.evaluate_detector(
                model,
                dl_val,
                ANCHORS,
                img_size=(640, 320),
                grid_size=(40, 20),
                conf_th=0.20,    
                nms_iou=0.01,
                match_iou=0.5,
                num_classes=7,
                max_images=200,      
                device=device,
            )
            print(
                f"[VAL] epoch {epoch:03d} "
                f"P={metrics['precision']:.3f} "
                f"R={metrics['recall']:.3f} "
                f"mAP50={metrics['mAP50']:.3f} "
                f"F1 = {(2 *metrics['recall'] * metrics['precision']) / (metrics['recall'] + metrics['precision'])}"
            )
            model.train()


    model.eval()
    export.export_all_outputs(model)

if __name__ == "__main__":
    main()


