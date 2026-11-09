**English** | [简体中文](README.zh-CN.md)

# Manga Panel Detection — RF-DETR (trained on Manga109-s)

Single-class comic **panel (frame) detector** for manga pages, built on
[RF-DETR](https://github.com/roboflow/rf-detr) (Apache-2.0). DETR-family
set prediction — **no NMS needed**. Designed for page-by-page comic reading
guides in the open-source Android reader **Mandu (漫读)**.

## Results

| Benchmark | This model | Baseline (forkger weights) |
|---|---|---|
| Full Manga109-s spreads (8147 pages, greedy IoU≥0.5, micro F1) | **0.947** (P 0.952 / R 0.943) | 0.944 |
| Internal 13-page reader benchmark (real reader pages, corrected GT)¹ | **0.917** | 0.890 |

**How to read these numbers**: the two models share the same recipe and land
within measurement noise of each other on the full evaluation — a 0.3-point
margin can flip with threshold or harness details. This comparison exists to
validate that my training pipeline reproduces the recipe correctly, not to
rank the two models; a baseline this strong is exactly why the recipe's
author is credited above. The 13-page margin (2.7 points, n = 13 pages) is a
small sample — read it as a hint of better robustness on hard cases, not a
verdict.

On-device latency (Android, FP16 ONNX, onnxruntime 1.31.1, 4 CPU threads,
Snapdragon 8+): **~1.1 s per page** at 576×576.

In Mandu the overlay is fully async: the page renders immediately and boxes
arrive when inference finishes; the next page is prefetched during sequential
reading, so in day-to-day panel-by-panel use the 1.1 s is mostly invisible.
The runtime ships all Android ABIs (32-bit included) — no arm64-only gap.

¹ The 13-page benchmark is a small hand-built stress test, not a random
sample: real reader pages from commercially published manga volumes, hand-annotated,
and deliberately weighted toward the cases that are hard to split in
practice — bubbles crossing panel gutters, slanted panel layouts. Few
pages, but exactly where engines actually fail in the field. The pages
are copyrighted and the GT is **not** redistributed, so it is not
reproducible from this repo; the full Manga109-s evaluation above is —
see `eval/`.

## Model

| | |
|---|---|
| Files | `selftrain-fp16.onnx` (57 MB, mobile) · `selftrain-fp32.onnx` (114 MB, master) · `checkpoint_best_ema.pth` (134 MB, for fine-tuning) — see [Releases](../../releases) |
| Variant | RF-DETR **Medium @ 576** |
| Input | `float32 [1,3,576,576]`, **CHW** (channel-planar), ImageNet mean/std, full-page stretch (no letterbox) |
| Orientation | Portrait single pages: as-is. **Landscape double-page spreads: rotate 90° clockwise** (`PIL rotate(-90)`) before resize |
| Output `dets` | `[1,300,4]` — (cx, cy, w, h), normalized 0–1 |
| Output `labels` | `[1,300,2]` — raw logits; **sigmoid(col 0) = panel confidence** (col 1 = background); no NMS; threshold 0.40 standard / 0.25 loose / 0.55 strict |

Machine-readable tensor spec: [`onnx_io.json`](onnx_io.json).

```python
import numpy as np
from PIL import Image
import onnxruntime as ort

sess = ort.InferenceSession("selftrain-fp16.onnx", providers=["CPUExecutionProvider"])
im = Image.open("page.jpg").convert("RGB")
if im.width > im.height:                       # landscape spread -> rotate
    im = im.rotate(-90, expand=True)
x = np.asarray(im.resize((576, 576), Image.BILINEAR), np.float32) / 255.0
x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
dets, labels = sess.run(None, {"input": x.transpose(2, 0, 1)[None]})
score = 1 / (1 + np.exp(-labels[0, :, 0]))      # sigmoid -> panel confidence
boxes = dets[0][score > 0.40]                   # cxcywh, normalized
```

## Android notes (pitfalls I hit, so you don't have to)

- **R8/ProGuard**: release builds with minification must keep
  `-keep class ai.onnxruntime.** { *; }`. R8 deletes `ai.onnxruntime.TensorInfo`
  (nothing references it directly) and the JNI does `FindClass` by name when
  building outputs → `java_class == null in call to GetMethodID` abort,
  release-only. Same as [onnxruntime#17847](https://github.com/microsoft/onnxruntime/issues/17847).
- **CHW, not HWC**: feeding HWC-interleaved pixels into the `[1,3,576,576]`
  tensor **silently yields 0 detections** — no crash, no error.
- **Skip INT8**: dynamic quantization produces `ConvInteger`, which has no
  ORT CPU kernel. FP16 (`onnxruntime.transformers.float16`) is the practical
  mobile path — zero quality loss measured (INT8 lost ~1.5 F1 points anyway).

## Training

- Data: [Manga109-s](http://www.manga109.org) (8533 double-page spreads)
- Recipe: RF-DETR **Medium @ 576**, 50 epochs, EMA best (val ema_mAP50 =
  0.987 at epoch 35), lr 1e-4, batch 4 × grad-accum 4 — same recipe as
  [forkger/manga-panel-rfdetr](https://github.com/forkger/manga-panel-rfdetr)
- Export: ONNX (stretch + ImageNet preprocessing — letterbox measured worse)
  → FP16 via `onnxruntime.transformers.float16`
- Per-epoch validation curve: [`training/results.csv`](training/results.csv)

## Special Thanks

**[forkger](https://github.com/forkger)** ([manga-panel-rfdetr](https://github.com/forkger/manga-panel-rfdetr)).

This model started from his released work: the training recipe
(RF-DETR Medium @ 576, 50 epochs, EMA) is his, and his eval notes are
the reason I skipped letterbox preprocessing and INT8 quantization instead
of wasting weeks on both. He also answered every question I asked along
the way. If this model is useful to you, star his repo.

**中文**：特别感谢 [forkger](https://github.com/forkger)。训练配方（RF-DETR
Medium @ 576、50 epochs、EMA）直接沿用其公开发布的工作；放弃 letterbox
预处理和 INT8 量化这两个决定，也来自他的评测笔记；过程中我提的问题他都
认真答了。如果这个模型对你有用，请去给他的仓库点个 star。

## License & Acknowledgements

- Code and model weights: **Apache-2.0** (see `LICENSE`).
- **This model is trained on [Manga109-s](http://www.manga109.org)** — use of
  the dataset follows its terms; dataset images are **not** redistributed
  here. Please cite the Manga109-s paper if you use this model in research,
  and retain the "trained on Manga109-s" attribution when redistributing.
- Architecture: [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow (Apache-2.0).
- Runtime: [onnxruntime](https://github.com/microsoft/onnxruntime) (MIT).
