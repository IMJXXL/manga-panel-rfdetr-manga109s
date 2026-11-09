[English](README.md) | **简体中文**

# 漫画分格检测 — RF-DETR（Manga109-s 训练）

单类漫画**分格（画格）检测模型**，基于 [RF-DETR](https://github.com/roboflow/rf-detr)（Apache-2.0）。DETR 集合预测，**免 NMS**。为开源安卓漫画阅读器「漫读」的逐格阅读引导而训练。

## 评测结果

| 基准 | 本模型 | 基线（forkger 权重） |
|---|---|---|
| Manga109-s 全量横页（8147 页，贪心 IoU≥0.5，微平均 F1） | **0.947**（P 0.952 / R 0.943） | 0.944 |
| 内部 13 页实战基准（真实阅读页，修正后 GT）¹ | **0.917** | 0.890 |

真机延迟（安卓，FP16 ONNX，onnxruntime 1.31.1，4 CPU 线程，骁龙 8+）：576×576 单页约 **1.1 秒**。

「漫读」的覆盖层完全异步：页面立即渲染，检测完成后框才浮现，顺序阅读时下一页自动预取——日常逐格阅读基本感觉不到这 1.1 秒。运行时带全 ABI（含 32 位），没有 TFLite 那种 arm64-only 的缺口。

¹ 13 页快尺是小型手工压力测试，不是随机抽样：真实阅读页（商业出版的漫画单行本），手工标注，刻意加重了实战里最难分的情形——跨界气泡、斜格分格。页数少，但都是引擎实战真正会翻车的地方。页面有版权、GT 不再分发，因此无法从本仓库复现；上面的全量评测可复现——见 `eval/`。

**这组数字怎么读**：两版模型同一配方，全量 0.3 个点的差距在测量噪声内——这张表是验证我的训练管线没有练坏，不是给两个模型排座次；基线能这么强，恰恰是配方作者厉害（见特别感谢）。13 页的 2.7 个点样本更小，当参考看。

## 模型

| | |
|---|---|
| 文件 | `selftrain-fp16.onnx`（57 MB，移动端）· `selftrain-fp32.onnx`（114 MB，主档）· `checkpoint_best_ema.pth`（134 MB，微调用）——见 [Releases](../../releases) |
| 规格 | RF-DETR **Medium @ 576** |
| 输入 | `float32 [1,3,576,576]`，**CHW**（通道平面），ImageNet mean/std，整页拉伸（无 letterbox） |
| 方向 | 竖页单页：直接推理。**横页双页拼版：先顺时针旋转 90°**（`PIL rotate(-90)`）再缩放 |
| 输出 `dets` | `[1,300,4]` —— (cx, cy, w, h)，归一化 0–1 |
| 输出 `labels` | `[1,300,2]` —— 原始 logits；**sigmoid(col 0) = 分格置信度**（col 1 = 背景）；免 NMS；阈值 0.40 标准 / 0.25 宽松 / 0.55 严格 |

机器可读张量规格：[`onnx_io.json`](onnx_io.json)。

```python
import numpy as np
from PIL import Image
import onnxruntime as ort

sess = ort.InferenceSession("selftrain-fp16.onnx", providers=["CPUExecutionProvider"])
im = Image.open("page.jpg").convert("RGB")
if im.width > im.height:                       # 横页拼版 -> 旋转
    im = im.rotate(-90, expand=True)
x = np.asarray(im.resize((576, 576), Image.BILINEAR), np.float32) / 255.0
x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
dets, labels = sess.run(None, {"input": x.transpose(2, 0, 1)[None]})
score = 1 / (1 + np.exp(-labels[0, :, 0]))      # sigmoid -> 分格置信度
boxes = dets[0][score > 0.40]                   # cxcywh，归一化
```

## 安卓避坑笔记（我踩过的，你不必再踩）

- **R8/ProGuard**：release 混淆包必须加 `-keep class ai.onnxruntime.** { *; }`。R8 会删掉 `ai.onnxruntime.TensorInfo`（没有任何代码直接引用它），而 JNI 构造输出时按原始类名 `FindClass` → `java_class == null in call to GetMethodID` abort，**只在 release 包出现**。同 [onnxruntime#17847](https://github.com/microsoft/onnxruntime/issues/17847)。
- **CHW，不是 HWC**：把 HWC 交错的像素灌进 `[1,3,576,576]` 张量会**静默输出 0 框**——不崩溃、不报错。
- **别碰 INT8**：动态量化产生 `ConvInteger`，ORT CPU 没有这个内核。FP16（`onnxruntime.transformers.float16`）才是移动端正路——实测零精度损失（INT8 本身还掉了约 1.5 个 F1 点）。

## 训练

- 数据：[Manga109-s](http://www.manga109.org)（8533 张双页拼版）
- 配方：RF-DETR **Medium @ 576**，50 epochs，EMA best（val ema_mAP50 = 0.987，epoch 35），lr 1e-4，batch 4 × 梯度累积 4——与 [forkger/manga-panel-rfdetr](https://github.com/forkger/manga-panel-rfdetr) 同配方
- 导出：ONNX（stretch + ImageNet 预处理——letterbox 实测更差）→ `onnxruntime.transformers.float16` 转 FP16
- 逐 epoch 验证曲线：[`training/results.csv`](training/results.csv)

## 特别感谢

**[forkger](https://github.com/forkger)**（[manga-panel-rfdetr](https://github.com/forkger/manga-panel-rfdetr)）。

这个模型从他的公开工作起步：训练配方（RF-DETR Medium @ 576、50 epochs、EMA）是他的，放弃 letterbox 预处理和 INT8 量化这两个决定也来自他的评测笔记——不然这两条路我要白白耗掉几周。过程中我提的每一个问题他都认真答了。如果这个模型对你有用，请去给他的仓库点个 star。

## 许可与致谢

- 代码与模型权重：**Apache-2.0**（见 `LICENSE`）。
- **本模型基于 [Manga109-s](http://www.manga109.org) 训练**——数据集使用遵循其条款；数据集图片**不**在本仓库再分发。研究用途请引用 Manga109-s 论文，再分发时请保留 "trained on Manga109-s" 署名。
- 模型架构：Roboflow 的 [RF-DETR](https://github.com/roboflow/rf-detr)（Apache-2.0）。
- 运行时：[onnxruntime](https://github.com/microsoft/onnxruntime)（MIT）。
