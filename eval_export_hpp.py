# # -*- coding: utf-8 -*-
# """eval_export_hpp.py

# 用途：
#   - 從 export_out/*.hpp（例如 weights.hpp）讀回 HLS/FPGA 版本的參數
#   - 建立同一份 QAT 模型 (structure.HLSLikeQATDetector)
#   - 呼叫 evaluate.evaluate_detector() 跑 mAP/precision/recall


from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

# 讓你從任何 working directory 執行，都能 import 同資料夾下的 train/evaluate/structure
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import torch
from torch.utils.data import DataLoader, Subset

import structure
import train
import evaluate
from champ_weight_loader import load_champ_weights_into_model


# === 必須跟 HLS / train.py 對齊的 config ===
SHIFTS = [19, 15, 15, 15, 15, 15, 15, 15]
INC_BITS_NEW = [15, 13, 12, 11, 12, 12, 12, 12]
BIAS_BITS_NEW = [25, 21, 21, 21, 21, 21, 21, 21]
HEAD_BIAS_BITS_NEW = 11


def build_model(device: str) -> torch.nn.Module:
    model = structure.HLSLikeQATDetector(
        shifts=SHIFTS,
        inc_bits_new=INC_BITS_NEW,
        bias_bits_new=BIAS_BITS_NEW,
        head_bias_bits_new=HEAD_BIAS_BITS_NEW,
    ).to(device)
    return model


def build_val_loader(
    img_dir: str,
    label_dir: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    use_all: bool,
) -> tuple[DataLoader, int]:
    ds_all = train.JsonDetDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=(640, 320),
    )

    if use_all or val_ratio <= 0.0:
        ds_val = ds_all
        n_val = len(ds_all)
    else:
        idxs = list(range(len(ds_all)))
        random.Random(seed).shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        ds_val = Subset(ds_all, idxs[:n_val])

    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=train.collate_fn,
        pin_memory=True,
    )
    return dl_val, n_val


def prf1_from_metrics(metrics: dict) -> tuple[float, float, float]:
    """Return (precision, recall, f1)."""
    p = float(metrics.get("precision", 0.0))
    r = float(metrics.get("recall", 0.0))
    f1 = 0.0 if (p + r) <= 1e-12 else (2.0 * p * r / (p + r))
    return p, r, f1


@torch.no_grad()
def eval_prf1_with_per_class(
    model: torch.nn.Module,
    dl_val: DataLoader,
    *,
    anchors,
    img_size=(640, 320),
    grid_size=(40, 20),
    conf_th: float = 0.25,
    nms_iou: float = 0.45,
    match_iou: float = 0.5,
    num_classes: int = 7,
    max_images: int | None = None,
    max_det: int = 5,
    device: str = "cuda",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Compute global P/R/F1 and per-class precision at the chosen operating point.

    注意：這裡的 *per-class precision* 指的是：
      precision_c = TP_c / (TP_c + FP_c)
    其中 TP/FP 都是在你的 postprocess(conf_th + NMS + max_det) 之後，
    以 match_iou 做一對一 matching 得到的結果。
    """

    model.eval()

    tp_c = [0 for _ in range(num_classes)]
    fp_c = [0 for _ in range(num_classes)]
    gt_c = [0 for _ in range(num_classes)]

    total_tp = 0
    total_fp = 0
    total_gt = 0

    seen = 0

    it = dl_val
    if show_progress:
        try:
            from tqdm.auto import tqdm  # local import (避免 tqdm 不存在時掛掉)

            it = tqdm(dl_val, desc="Val", leave=False)
        except Exception:
            it = dl_val

    for imgs, boxes_list, labels_list, _ in it:
        imgs = imgs.to(device, non_blocking=True)
        y_pred = model(imgs)

        # decode + NMS (與 evaluate.py 對齊)
        results = train.postprocess(
            y_pred,
            anchors,
            img_size=img_size,
            grid_size=grid_size,
            conf_th=conf_th,
            iou_th=nms_iou,
            max_det=max_det,
        )

        B = imgs.size(0)
        for b in range(B):
            gt_boxes = boxes_list[b].to(device)
            gt_labels = labels_list[b].to(device)

            total_gt += int(gt_boxes.shape[0])
            for c in range(num_classes):
                gt_c[c] += int((gt_labels == c).sum().item())

            # group gt by class
            gt_by_cls = [gt_boxes[gt_labels == c] for c in range(num_classes)]

            det_boxes = results[b]["boxes"].to(device)
            det_scores = results[b]["scores"].to(device)
            det_cls = results[b]["classes"].to(device)

            # match per class (與 evaluate.py 同一套規則)
            for c in range(num_classes):
                mc = (det_cls == c)
                if mc.sum() == 0:
                    continue

                db = det_boxes[mc]
                ds = det_scores[mc]
                order = torch.argsort(ds, descending=True)
                db = db[order]
                ds = ds[order]

                gt_boxes_c = gt_by_cls[c]
                n_gt = int(gt_boxes_c.shape[0])
                matched = torch.zeros((n_gt,), dtype=torch.bool, device=device)

                for k in range(int(db.shape[0])):
                    if n_gt == 0:
                        fp_c[c] += 1
                        total_fp += 1
                        continue

                    ious = train.box_iou_xyxy(db[k].unsqueeze(0), gt_boxes_c).squeeze(0)
                    best_iou, best_j = torch.max(ious, dim=0)
                    if best_iou.item() >= match_iou and (not matched[best_j].item()):
                        matched[best_j] = True
                        tp_c[c] += 1
                        total_tp += 1
                    else:
                        fp_c[c] += 1
                        total_fp += 1

            seen += 1
            if max_images is not None and seen >= max_images:
                break
        if max_images is not None and seen >= max_images:
            break

    p = float(total_tp / (total_tp + total_fp + 1e-9))
    r = float(total_tp / (total_gt + 1e-9))
    f1 = 0.0 if (p + r) <= 1e-12 else float(2.0 * p * r / (p + r))

    per_class_precision: list[float | None] = []
    for c in range(num_classes):
        denom = tp_c[c] + fp_c[c]
        per_class_precision.append(None if denom == 0 else float(tp_c[c] / (denom + 1e-9)))

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp_per_class": tp_c,
        "fp_per_class": fp_c,
        "gt_per_class": gt_c,
        "precision_per_class": per_class_precision,
        "images": seen,
    }


def print_prf1(p: float, r: float, f1: float):
    # 依你要求：每個模型只印 P/R/F1
    print(f"precision = {p:.4f}")
    print(f"recall    = {r:.4f}")
    print(f"f1        = {f1:.4f}")


def format_per_class_precision(prec_pc: list[float | None]) -> str:
    parts = []
    for i, v in enumerate(prec_pc):
        if v is None:
            parts.append(f"c{i}:NA")
        else:
            parts.append(f"c{i}:{v:.4f}")
    return " ".join(parts)


def one_line_prf1(p: float, r: float, f1: float) -> str:
    """Compact summary for printing in a table."""
    return f"P={p:.4f}  R={r:.4f}  F1={f1:.4f}"


def resolve_hpp_list(weights_hpp: str | None, weights_dir: str | None, weights_glob: str) -> list[Path]:
    paths: list[Path] = []
    if weights_hpp:
        p = Path(weights_hpp)
        paths.append(p)
    if weights_dir:
        d = Path(weights_dir)
        if d.is_dir():
            paths.extend(sorted(d.glob(weights_glob)))
        else:
            raise FileNotFoundError(d)

    # 如果使用者都沒給，就預設 THIS_DIR/export_out/*.hpp
    if not paths:
        d = THIS_DIR / "export_out"
        if d.is_dir():
            paths = sorted(d.glob(weights_glob))

    # 去重（保留順序）
    seen = set()
    uniq: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights_hpp", type=str, default=None, help="export_out/weights.hpp (or other .hpp) path")
    p.add_argument("--weights_dir", type=str, default=None, help="directory containing .hpp files (default: ./export_out)")
    p.add_argument("--weights_glob", type=str, default="*.hpp", help="glob pattern under --weights_dir (default: *.hpp)")
    p.add_argument("--state_dict", type=str, default=None, help="export_out/qat_model_state.pth path")

    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--label_dir", type=str, required=True)

    p.add_argument("--batch", type=int, default=16)
    # Windows 建議 0；Linux 可以開到 4~8
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--val_ratio", type=float, default=0.1, help="split ratio used for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all", action="store_true", help="evaluate on ALL images (ignore val_ratio)")

    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--nms_iou", type=float, default=0.01, help="NMS IoU threshold")
    p.add_argument("--match_iou", type=float, default=0.5, help="IoU threshold for TP matching")
    p.add_argument("--max_det", type=int, default=5, help="postprocess max_det (must match evaluate.py/train.postprocess)")

    p.add_argument("--max_images", type=int, default=None, help="debug: limit evaluation images")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="force cpu/cuda")

    p.add_argument("--per_class", action="store_true", help="also print per-class precision (c0..c6)")
    p.add_argument("--per_class_detail", action="store_true", help="when using --per_class, also print tp/fp/gt for each class")
    p.add_argument("--no_progress", action="store_true", help="disable tqdm progress bar")

    p.add_argument("--sort", type=str, default="f1", choices=["f1", "precision", "recall", "name"], help="sort summary table")
    p.add_argument("--descending", action="store_true", help="sort descending (default True for metrics)")

    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # 1) dataloader（所有 weights 共用）
    dl_val, n_val = build_val_loader(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        batch_size=args.batch,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_all=args.all,
    )
    print(f"[INFO] val_images = {n_val}  batch={args.batch}  num_workers={args.num_workers}")

    # 2) 如果是 state_dict：只跑一次
    if args.state_dict is not None:
        spath = Path(args.state_dict)
        if not spath.exists():
            raise FileNotFoundError(spath)
        model = build_model(device)
        sd = torch.load(str(spath), map_location=device)
        model.load_state_dict(sd, strict=True)
        print(f"[OK] loaded state_dict: {spath}")

        out = eval_prf1_with_per_class(
            model,
            dl_val,
            anchors=train.ANCHORS,
            img_size=(640, 320),
            grid_size=(40, 20),
            conf_th=args.conf,
            nms_iou=args.nms_iou,
            match_iou=args.match_iou,
            num_classes=7,
            max_images=args.max_images,
            max_det=args.max_det,
            device=device,
	            show_progress=not args.no_progress,
        )
        print_prf1(out["precision"], out["recall"], out["f1"])
        if args.per_class:
            print("per_class_precision:", format_per_class_precision(out["precision_per_class"]))
            if args.per_class_detail:
                for c in range(7):
                    print(f"  c{c}: tp={out['tp_per_class'][c]} fp={out['fp_per_class'][c]} gt={out['gt_per_class'][c]}")
        return

    # 3) .hpp：支援一次跑很多個
    hpps = resolve_hpp_list(args.weights_hpp, args.weights_dir, args.weights_glob)
    if not hpps:
        raise SystemExit("No .hpp found. Provide --weights_hpp or --weights_dir (or put files under ./export_out)")

    print(f"[INFO] found {len(hpps)} hpp file(s)")

    results: list[dict[str, Any]] = []
    for i, wpath in enumerate(hpps, 1):
        wpath = Path(wpath)
        if not wpath.exists():
            print(f"[{i}/{len(hpps)}] [SKIP] not found: {wpath}")
            results.append({"name": wpath.name, "path": str(wpath), "ok": False, "error": "not found"})
            continue

        print(f"\n[{i}/{len(hpps)}] ===== {wpath.name} =====")
        try:
            model = build_model(device)
            load_champ_weights_into_model(model, str(wpath))
            model.eval()

            out = eval_prf1_with_per_class(
                model,
                dl_val,
                anchors=train.ANCHORS,
                img_size=(640, 320),
                grid_size=(40, 20),
                conf_th=args.conf,
                nms_iou=args.nms_iou,
                match_iou=args.match_iou,
                num_classes=7,
                max_images=args.max_images,
                max_det=args.max_det,
                device=device,
	                show_progress=not args.no_progress,
            )

            p_val, r_val, f1_val = out["precision"], out["recall"], out["f1"]
            print_prf1(p_val, r_val, f1_val)
            if args.per_class:
                print("per_class_precision:", format_per_class_precision(out["precision_per_class"]))
                if args.per_class_detail:
                    for c in range(7):
                        print(f"  c{c}: tp={out['tp_per_class'][c]} fp={out['fp_per_class'][c]} gt={out['gt_per_class'][c]}")
            results.append({
                "name": wpath.name,
                "path": str(wpath),
                "ok": True,
                "precision": p_val,
                "recall": r_val,
                "f1": f1_val,
                "precision_per_class": out["precision_per_class"],
                "tp_per_class": out["tp_per_class"],
                "fp_per_class": out["fp_per_class"],
                "gt_per_class": out["gt_per_class"],
            })
        except Exception as e:
            print(f"[FAIL] {wpath.name}: {type(e).__name__}: {e}")
            results.append({"name": wpath.name, "path": str(wpath), "ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()

    # 4) summary
    print("\n\n================ SUMMARY ================")
    ok_rows = [r for r in results if r.get("ok")]
    fail_rows = [r for r in results if not r.get("ok")]

    # sort
    sort_key = args.sort
    if sort_key == "name":
        keyfn = lambda r: r.get("name", "")
        desc = False
    else:
        keyfn = lambda r: float(r.get(sort_key, 0.0))
        desc = True
    if args.descending:
        desc = True
    ok_rows = sorted(ok_rows, key=keyfn, reverse=desc)

    if ok_rows:
        print(f"OK: {len(ok_rows)}/{len(results)}")
        print("name\tprecision\trecall\tf1")
        for r in ok_rows:
            print(f"{r['name']}\t{r['precision']:.4f}\t{r['recall']:.4f}\t{r['f1']:.4f}")
    else:
        print("OK: 0")

    if fail_rows:
        print(f"\nFAIL: {len(fail_rows)}/{len(results)}")
        for r in fail_rows:
            print(f"- {r.get('name')}: {r.get('error')}")


if __name__ == "__main__":
    main()
