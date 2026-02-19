import numpy as np
import torch
from tqdm.auto import tqdm
import train


def compute_ap_voc(rec, prec):
    """
    VOC-style AP with precision envelope.
    rec, prec: 1D numpy arrays (increasing rec)
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # precision envelope
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # area under PR curve
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

@torch.no_grad()
def evaluate_detector(
    model, dl_val, anchors,
    img_size=(640,320), grid_size=(40,20),
    conf_th=0.25, nms_iou=0.45,
    match_iou=0.5,
    num_classes=7,
    max_images=None,
    device="cuda"
):
    """
    回傳:
      metrics = {
        "precision": float,
        "recall": float,
        "mAP50": float,
        "AP50_per_class": list[float],
        "num_gt_per_class": list[int],
      }
    """
    model.eval()

    # 每類別累積：scores, tps, fps, gt_count
    all_scores = [[] for _ in range(num_classes)]
    all_tps    = [[] for _ in range(num_classes)]
    all_fps    = [[] for _ in range(num_classes)]
    gt_counts  = [0 for _ in range(num_classes)]

    total_tp = 0
    total_fp = 0
    total_gt = 0

    seen = 0

    total_pred = 0
    min_pred = 10**9
    max_pred = 0


    
    for imgs, boxes_list, labels_list, _ in tqdm(dl_val, desc="Val", leave=False):

        imgs = imgs.to(device, non_blocking=True)
        y_pred = model(imgs)  # (B,72,20,40)

        # decode + softmax + NMS
        results = train.postprocess(
            y_pred, anchors,
            img_size=img_size, grid_size=grid_size,
            conf_th=conf_th, iou_th=nms_iou, max_det=5
        )


        B = imgs.size(0)
        for b in range(B):
            gt_boxes = boxes_list[b].to(device)    # (Ng,4)
            gt_labels = labels_list[b].to(device)  # (Ng,)

            # GT per class
            gt_by_cls = []
            for c in range(num_classes):
                m = (gt_labels == c)
                gt_by_cls.append(gt_boxes[m])
                gt_counts[c] += int(m.sum().item())

            total_gt += int(gt_boxes.shape[0])

            # Dets from postprocess (already cpu) -> to device
            det_boxes = results[b]["boxes"].to(device)      # (Nd,4)
            det_scores = results[b]["scores"].to(device)    # (Nd,)
            det_cls = results[b]["classes"].to(device)      # (Nd,)

            Nd = int(det_boxes.shape[0])  # number of predicted boxes for this image (<= max_det=8)
            total_pred += Nd
            if Nd < min_pred: min_pred = Nd
            if Nd > max_pred: max_pred = Nd


            # per class matching
            for c in range(num_classes):
                # detections of class c, sorted by score desc
                mc = (det_cls == c)
                if mc.sum() == 0:
                    continue

                db = det_boxes[mc]
                ds = det_scores[mc]
                order = torch.argsort(ds, descending=True)
                db = db[order]
                ds = ds[order]

                gt_c = gt_by_cls[c]
                n_gt = gt_c.shape[0]
                matched = torch.zeros((n_gt,), dtype=torch.bool, device=device)

                for k in range(db.shape[0]):
                    score_k = float(ds[k].item())
                    if n_gt == 0:
                        # no gt -> all FP
                        all_scores[c].append(score_k)
                        all_tps[c].append(0)
                        all_fps[c].append(1)
                        total_fp += 1
                        continue

                    ious = train.box_iou_xyxy(db[k].unsqueeze(0), gt_c).squeeze(0)  # (n_gt,)
                    best_iou, best_j = torch.max(ious, dim=0)
                    if best_iou.item() >= match_iou and (not matched[best_j].item()):
                        matched[best_j] = True
                        all_scores[c].append(score_k)
                        all_tps[c].append(1)
                        all_fps[c].append(0)
                        total_tp += 1
                    else:
                        all_scores[c].append(score_k)
                        all_tps[c].append(0)
                        all_fps[c].append(1)
                        total_fp += 1

            # early stop
            seen += 1
            if max_images is not None and seen >= max_images:
                break

        if max_images is not None and seen >= max_images:
            break

    # precision/recall overall
    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall    = total_tp / (total_gt + 1e-9)

    # AP per class
    ap_list = []
    valid_aps = []
    for c in range(num_classes):
        n_gt = gt_counts[c]
        if len(all_scores[c]) == 0:
            ap = 0.0
        else:
            scores = np.array(all_scores[c], dtype=np.float32)
            tps = np.array(all_tps[c], dtype=np.float32)
            fps = np.array(all_fps[c], dtype=np.float32)

            # sort by score desc
            idx = np.argsort(-scores)
            tps = tps[idx]
            fps = fps[idx]

            tp_cum = np.cumsum(tps)
            fp_cum = np.cumsum(fps)

            if n_gt == 0:
                ap = 0.0
            else:
                prec = tp_cum / (tp_cum + fp_cum + 1e-9)
                rec  = tp_cum / (n_gt + 1e-9)
                ap = compute_ap_voc(rec, prec)

        ap_list.append(ap)
        if gt_counts[c] > 0:
            valid_aps.append(ap)

    mAP50 = float(np.mean(valid_aps)) if len(valid_aps) > 0 else 0.0

    if seen > 0:
        avg_pred = total_pred / seen
        if min_pred == 10**9: min_pred = 0
    else:
        avg_pred, std_pred, min_pred, max_pred = 0.0, 0.0, 0, 0

    print(f"[VAL] avg_pred_boxes/img (max_det=5) = {avg_pred:.3f}  "
        f"min={min_pred}, max={max_pred}, images={seen})")


    return {
        "precision": float(precision),
        "recall": float(recall),
        "mAP50": mAP50,
        "AP50_per_class": ap_list,
        "num_gt_per_class": gt_counts
    }
