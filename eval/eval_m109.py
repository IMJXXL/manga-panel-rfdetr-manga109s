"""Full Manga109-s evaluation for RF-DETR manga-panel ONNX models.

Reproduces the headline numbers of this repo: greedy one-to-one matching at
IoU >= 0.5, micro-averaged (P/R/F1) over all annotated frames.

Usage:
    # this repo's model (FP16, ImageNet norm, 2-col labels)
    python eval/eval_m109.py \
        --dataset-root /path/to/Manga109s_released_XXX \
        --onnx selftrain-fp16.onnx --norm imagenet --conf 0.40

    # baseline comparison (forkger weights: div255 norm; 5-col labels auto-detected)
    python eval/eval_m109.py --dataset-root ... --onnx <forkger>.onnx --norm div255

The dataset itself is NOT redistributed here. Get it (research registration)
at http://www.manga109.org and point --dataset-root at the extracted folder
(containing images/ and annotations*/).

Notes:
  - Landscape double-page spreads are rotated 90 degrees clockwise
    (PIL rotate(-90)) before inference, matching the training-time
    convention; ground-truth frames are rotated per-box by a pixel
    round-trip to stay in the rotated coordinate space.
  - Resume-safe: pages already present in the output CSV are skipped.
"""
import argparse, csv, glob, os, time
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort

RES = 576


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(gt, det):
    pairs = sorted(((iou(g, d), gi, di) for gi, g in enumerate(gt) for di, d in enumerate(det)), reverse=True)
    ug, ud = set(), set()
    for v, gi, di in pairs:
        if v < 0.5 or gi in ug or di in ud:
            continue
        ug.add(gi); ud.add(di)
    return len(ug)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True,
                    help="extracted Manga109-s release folder (contains images/ and annotations*/)")
    ap.add_argument("--onnx", required=True, help="model ONNX path")
    ap.add_argument("--norm", default="imagenet", choices=["div255", "imagenet"],
                    help="input normalization: imagenet (this repo's model) / div255 (forkger weights)")
    ap.add_argument("--conf", type=float, default=0.40, help="panel confidence threshold (post-sigmoid)")
    ap.add_argument("--limit", type=int, default=0, help="limit number of pages (debug)")
    ap.add_argument("--out-csv", default="m109_results.csv", help="per-page result CSV (resume-safe)")
    args = ap.parse_args()

    root = args.dataset_root
    img_dir = os.path.join(root, "images")
    ann_cands = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d) and os.path.basename(d).startswith("annotations"))
    if not os.path.isdir(img_dir) or not ann_cands:
        raise SystemExit(f"dataset layout not found under {root}: expect images/ and annotations*/")
    xml_dir = ann_cands[0]

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    done = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv, encoding="utf-8") as f:
            done = {(r[0], r[1]) for r in csv.reader(f) if r and r[0] != "book"}
    out_f = open(args.out_csv, "a", newline="", encoding="utf-8")
    writer = csv.writer(out_f)
    if not done:
        writer.writerow(["book", "page", "gt", "det", "match"])

    books = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(xml_dir, "*.xml")))
    todo = []
    for book in books:
        tree = ET.parse(os.path.join(xml_dir, book + ".xml")).getroot()
        for page in tree.iter("page"):
            frames = page.findall("frame")
            if not frames:
                continue
            key = (book, page.get("index"))
            if key in done:
                continue
            pw_, ph_ = int(page.get("width")), int(page.get("height"))
            todo.append((book, page.get("index"), pw_, ph_,
                         [(float(fr.get("xmin")), float(fr.get("ymin")), float(fr.get("xmax")), float(fr.get("ymax"))) for fr in frames]))
    if args.limit:
        todo = todo[: args.limit]
    total_gt = sum(len(t[4]) for t in todo)
    print(f"pages to evaluate {len(todo)} / {total_gt} frames (skipping {len(done)} done)", flush=True)

    t_start = time.time()
    processed = 0
    for book, pidx, W, H, gt_norm in todo:
        img_path = os.path.join(img_dir, book, f"{int(pidx):03d}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, book, f"{int(pidx):03d}.png")
        if not os.path.exists(img_path):
            continue
        im_src = Image.open(img_path).convert("RGB")
        # Landscape spreads: rotate image 90 deg clockwise (PIL -90) and move each
        # GT frame into the rotated space by drawing it and reading back pixels.
        landscape = im_src.width > im_src.height
        W_src, H_src = im_src.size
        if landscape:
            imr_model = im_src.rotate(-90, expand=True)
            gt_rot = []
            for (gx0, gy0, gx1, gy1) in gt_norm:
                tmp = Image.new("RGB", (W_src, H_src), (0, 0, 0))
                dd = ImageDraw.Draw(tmp)
                dd.rectangle([gx0, gy0, gx1, gy1], outline=(0, 255, 0), width=12)
                tmpr = tmp.rotate(-90, expand=True)
                a = np.asarray(tmpr)
                g = (a[:, :, 1] > 180) & (a[:, :, 0] < 120) & (a[:, :, 2] < 120)
                ys, xs = np.where(g)
                if len(xs):
                    RW_, RH_ = tmpr.size
                    gt_rot.append((xs.min()/RW_, ys.min()/RH_, (xs.max()+1)/RW_, (ys.max()+1)/RH_))
            gt_norm = gt_rot
            im_src = imr_model
        im = im_src.resize((RES, RES), Image.BILINEAR)
        x = np.asarray(im, dtype=np.float32)
        if args.norm == "div255":
            x = x / 255.0
        elif args.norm == "imagenet":
            x = x / 255.0
            x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        outs = sess.run(None, {"input": x.transpose(2, 0, 1)[None]})
        dets = outs[0][0]
        labels = outs[1][0]
        # panel column auto-detected: 2-col labels = this repo's model (col 0);
        # 5-col labels = forkger baseline (col 1)
        panel_col = 1 if labels.shape[1] >= 5 else 0
        scores = 1.0 / (1.0 + np.exp(-labels[:, panel_col]))
        det_norm = []
        for q in range(300):
            if float(scores[q]) < args.conf:
                continue
            cx, cy, w, h = dets[q]
            det_norm.append((max(0.0, cx - w / 2), max(0.0, cy - h / 2),
                             min(1.0, cx + w / 2), min(1.0, cy + h / 2)))
        m = match(gt_norm, det_norm)
        writer.writerow([book, pidx, len(gt_norm), len(det_norm), m])
        processed += 1
        if processed % 200 == 0:
            out_f.flush()
            el = time.time() - t_start
            eta = el / processed * (len(todo) - processed)
            print(f"progress {processed}/{len(todo)}  {el/3600:.2f}h  ETA {eta/3600:.2f}h", flush=True)
    out_f.close()

    # summary
    rows = defaultdict(lambda: [0, 0, 0, 0])  # book -> [gt, det, match, pages]
    with open(args.out_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = rows[r["book"]]
            b[0] += int(r["gt"]); b[1] += int(r["det"]); b[2] += int(r["match"]); b[3] += 1
    T_gt = sum(v[0] for v in rows.values())
    T_det = sum(v[1] for v in rows.values())
    T_m = sum(v[2] for v in rows.values())
    P_ = T_m / T_det if T_det else 0
    R_ = T_m / T_gt if T_gt else 0
    F1 = 2 * P_ * R_ / (P_ + R_) if P_ + R_ else 0
    print("==== per-book (F1 ascending; low-scoring books = likely held-out volumes) ====")
    for book, (g, d, m, pg) in sorted(rows.items(), key=lambda kv: kv[1][2] / max(1, kv[1][0])):
        f1 = 2 * m / max(1, g + d)
        print(f"{book:24s} pages={pg:4d} gt={g:5d} det={d:5d} match={m:5d} F1={f1:.3f}")
    print(f"==== TOTAL: pages={sum(v[3] for v in rows.values())} gt={T_gt} det={T_det} match={T_m} P={P_:.3f} R={R_:.3f} F1={F1:.3f}")


if __name__ == "__main__":
    main()
