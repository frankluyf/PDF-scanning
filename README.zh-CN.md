# PDF-scanning

[English](README.md) | **简体中文**

一套完全在本地运行、GPU 加速的文档文字提取流水线，把金融类 PDF（借款合同、放款证明、还款记录）和身份证照片转换为纯文本，全程不把任何文件上传到云端 API。

它会**逐页**判断该用快速的文本层抽取还是视觉语言 OCR 模型：扫描页、金额表格这类关键页面走 OCR 保证准确，其余页面走文本层抽取保证速度。

> 状态：开发中，接口和目录结构可能还会调整。

---

## 为什么做这个

金融和身份类文件属于敏感数据，往往不能离开本机，所以云 OCR 服务用不了。而且这类文件质量参差不齐：

- 很多页面是电子生成的，自带干净的文本层，`pdfplumber` 几毫秒就能读完。
- 有些页面是扫描件或拍照，完全没有文本层。
- 含**金额表格**的页面（本金、利息、还款金额）最容易被文本层抽取搞错：单元格被合并、顺序被打乱、或者一行被拆成几行。这里错一位数字，代价很大。

每一页都跑 3B 参数的 OCR 模型，准确但慢；只用 `pdfplumber`，快但恰好在最关键的页面上出错。本项目的做法是把每一页分给最合适的工具。

## 工作原理

```
                 ┌──────────────────────────┐
  PDF 页面 ───▶  │ PyMuPDF：探测文本层        │
                 └────────────┬─────────────┘
                              │
        ┌─────────────────────┼──────────────────────────┐
        ▼                     ▼                          ▼
   文字少于 10 个        命中金额关键词 + 检测到表格      纯文本页
   （扫描/图片页）        （金额、本金、¥ …）
        │                     │                          │
        ▼                     ▼                          ▼
   VLM OCR（GPU）        VLM OCR（GPU）            pdfplumber（CPU）
                                                         │
                                               抽取为空？──▶ 兜底走 VLM OCR
```

1. **探测**：用 PyMuPDF（`fitz`）读取页面文本层。
2. **扫描页**：文字少于 10 个字符，说明是图片页，交给 OCR。
3. **金额表格页**：文本中出现任一金额相关关键词（`金额`、`大写`、`人民币`、`¥`、`本金`、`利息`、`还款` 等）时，才调用开销更大的 `find_tables()`；若检测到表格，交给 OCR 以保证金额准确。先做便宜的关键词判断，只有可能需要时才做慢的表格检测。
4. **纯文本页**：用 `pdfplumber` 抽取；如果结果为空，兜底转 OCR。
5. **OCR**：页面以 200 DPI 渲染成图片，交给本地视觉语言模型，使用 `document parsing` 提示词，输出 Markdown（表格依然保持表格结构）。每页处理完都会清理临时图片和模型附带生成的文件。

模型只**加载一次**，在多页、多文件之间复用（`process_single_pdf` 接收已加载的模型），结束时显式释放显存。

## 项目结构

| 文件 | 作用 |
| --- | --- |
| `main.py` | 入口。页面路由（`should_ocr_for_amount`）、单个 PDF 处理（`process_single_pdf`），以及单文件入口 `main()`：加载模型 → 处理 → 释放显存。 |
| `run_baidu_ocr_pdf.py` | OCR 后端：`load_baidu_ocr_model`、`process_page_with_ocr`（渲染页面 → 推理 → 读取 `result.md` → 清理），以及用于身份证照片等单张图片的 `process_single_image`。 |
| `run_pdfplumber.py` | 单页文本层快速抽取。 |
| `clean.py` | 运行前清空或创建输出目录。 |
| `run_baidu_ocr.py` | 用同一模型识别单张图片的独立脚本，方便测试。 |
| `run_got_ocr.py` | 早期 GOT-OCR 2.0 实验（`plain` / `format` 两种模式），保留作对比。 |
| `input/` | 示例输入（已加入 .gitignore）。 |
| `output_text/` | 提取结果输出（已加入 .gitignore）。 |

## 环境要求

- Python 3.10+
- 支持 CUDA 的 NVIDIA 显卡。开发时使用 8 GB 显存的 RTX 显卡，模型以 `bfloat16` 运行。
- OCR 模型的本地权重（通过 `transformers` 以 `trust_remote_code=True` 加载），例如放在 `D:\models\Unlimited-OCR`。可选：`run_got_ocr.py` 所需的 GOT-OCR 2.0 权重。

Python 依赖：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 按你的 CUDA 版本选择
pip install transformers PyMuPDF pdfplumber
```

> 目前 `requirements.txt` 只列了 `PyMuPDF` 和 `pdfplumber`，`torch` 和 `transformers` 需按上面的方式另行安装。

## 使用方法

修改 `main.py` 底部的配置：

```python
if __name__ == "__main__":
    MODEL_PATH = r"D:\models\Unlimited-OCR"
    TEST_PDF = r".\input\sample2.pdf"
    main(pdf_path=TEST_PDF, model_dir=MODEL_PATH)
```

然后运行：

```bash
python main.py
```

结果写入 `output_text/final_output.txt`，每页一个分段：

```
==================== Page 1 ====================

...提取出的文本或 Markdown...
```

控制台日志会显示每一页走了哪条路线（`📝` pdfplumber、`💰` 金额表格 → OCR、`⚠️` 扫描页 → OCR、`🚨` 兜底）。

### 批量处理

处理大量文件时，只加载一次模型，然后循环调用各个模块：

```python
from run_baidu_ocr_pdf import load_baidu_ocr_model, process_single_image
from main import process_single_pdf

model, tokenizer = load_baidu_ocr_model(r"D:\models\Unlimited-OCR")

process_single_pdf("contract.pdf", model, tokenizer, "out/contract.txt", "out")
text = process_single_image(model, tokenizer, "id_front.jpg", "out")
```

## 设计要点

- **隐私优先**：全部离线在本机运行，文件内容不出本机。
- **关键处保精度**：金额表格一律走 OCR，错一位数字的代价远高于多花几秒。
- **先便宜后昂贵**：关键词匹配在表格检测之前，表格检测在 OCR 之前，只有便宜的判断无法下结论时才执行下一步。
- **显存可控**：模型只加载一次，结束时用 `torch.cuda.empty_cache()` 显式释放，8 GB 消费级显卡即可运行。
- **每页状态干净**：OCR 模型会把结果写到固定的 `result.md`，每页读取后立即删除，避免页与页之间互相污染。

## 后续计划

- [ ] 用命令行参数替代硬编码路径
- [ ] 补全 `requirements.txt`（`torch`、`transformers`）
- [ ] 内置批量入口，支持整个文件夹的 PDF 和图片
- [ ] 关键词列表和文字数阈值可配置
- [ ] 输出结构化结果（JSON），便于下游字段抽取
