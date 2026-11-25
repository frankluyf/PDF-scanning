# PDF-scanning

**English** | [简体中文](README.zh-CN.md)

A fully local, GPU-accelerated pipeline that turns financial PDFs (loan contracts, disbursement certificates, repayment records) and ID-card photos into plain text, without sending any document to a cloud API.

It decides **page by page** whether to use a fast text-layer extractor or a vision-language OCR model, so you get OCR accuracy where it matters (scanned pages, amount tables) and text-layer speed everywhere else.

> Status: work in progress. Interfaces and file layout may still change.

---

## Why

Financial and identity documents are sensitive. They often can't leave the machine, so cloud OCR services are not an option. They are also mixed in quality:

- Many pages are digitally generated and have a clean text layer, which `pdfplumber` reads in milliseconds.
- Some pages are scans or photos with no text layer at all.
- Pages with **monetary tables** (principal, interest, repayment amounts) are where text-layer extraction most often fails, because cells get merged, reordered or split across lines. A mistake here is costly.

Running a 3B-parameter OCR model on every page is accurate but slow. Running only `pdfplumber` is fast but wrong on exactly the pages that matter. This project routes each page to the right tool.

## How it works

```
                 ┌──────────────────────────┐
  PDF page ───▶  │ PyMuPDF: probe text layer │
                 └────────────┬─────────────┘
                              │
        ┌─────────────────────┼──────────────────────────┐
        ▼                     ▼                          ▼
 < 10 chars of text    amount keyword + table      plain text page
 (scan / image page)   detected (金额, 本金, ¥ …)
        │                     │                          │
        ▼                     ▼                          ▼
   VLM OCR (GPU)         VLM OCR (GPU)            pdfplumber (CPU)
                                                         │
                                               empty result? ──▶ VLM OCR fallback
```

1. **Probe**: PyMuPDF (`fitz`) reads the page's text layer.
2. **Scanned page**: fewer than 10 characters means the page is an image, so it goes to OCR.
3. **Amount table page**: if the text contains any money-related keyword (`金额`, `大写`, `人民币`, `¥`, `本金`, `利息`, `还款`, …) the router runs the more expensive `find_tables()` check. If a table is found, the page goes to OCR to keep amounts accurate. The cheap keyword check runs first, so the slow table detection only runs on pages that might need it.
4. **Plain text page**: extracted with `pdfplumber`. If that returns nothing, the page falls back to OCR.
5. **OCR**: the page is rendered at 200 DPI and parsed by a local vision-language model with the `document parsing` prompt, which returns Markdown (tables stay as tables). Temporary images and model side-outputs are cleaned up after each page.

The model is loaded **once** and reused across pages and files (`process_single_pdf` takes an already-loaded model), then GPU memory is released explicitly at the end.

## Project structure

| File | Purpose |
| --- | --- |
| `main.py` | Entry point. Page router (`should_ocr_for_amount`), per-PDF processing (`process_single_pdf`) and the single-file `main()` that loads the model, processes, and frees GPU memory. |
| `run_baidu_ocr_pdf.py` | OCR backend: `load_baidu_ocr_model`, `process_page_with_ocr` (render page → infer → read `result.md` → clean up) and `process_single_image` for standalone images such as ID-card photos. |
| `run_pdfplumber.py` | Fast text-layer extraction for a single page. |
| `clean.py` | Clears or creates the output directory before a run. |
| `run_baidu_ocr.py` | Standalone script to OCR a single image with the same model. Useful for testing. |
| `run_got_ocr.py` | Earlier experiment with GOT-OCR 2.0 (`plain` / `format` modes), kept for comparison. |
| `input/` | Sample inputs (git-ignored). |
| `output_text/` | Extracted text output (git-ignored). |

## Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA. Development used an 8 GB RTX-class card; the model runs in `bfloat16`.
- Local model weights for the OCR model (loaded with `transformers` and `trust_remote_code=True`), for example under `D:\models\Unlimited-OCR`. Optional: GOT-OCR 2.0 weights for `run_got_ocr.py`.

Python packages:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick the build matching your CUDA
pip install transformers PyMuPDF pdfplumber
```

> `requirements.txt` currently lists only `PyMuPDF` and `pdfplumber`. `torch` and `transformers` must be installed as shown above.

## Usage

Edit the configuration at the bottom of `main.py`:

```python
if __name__ == "__main__":
    MODEL_PATH = r"D:\models\Unlimited-OCR"
    TEST_PDF = r".\input\sample2.pdf"
    main(pdf_path=TEST_PDF, model_dir=MODEL_PATH)
```

Then run:

```bash
python main.py
```

The result is written to `output_text/final_output.txt`, with one section per page:

```
==================== Page 1 ====================

...extracted text or Markdown...
```

The console log shows which route each page took (`📝` pdfplumber, `💰` amount table → OCR, `⚠️` scanned → OCR, `🚨` fallback).

### Batch processing

To process many files, load the model once and call the building blocks in a loop:

```python
from run_baidu_ocr_pdf import load_baidu_ocr_model, process_single_image
from main import process_single_pdf

model, tokenizer = load_baidu_ocr_model(r"D:\models\Unlimited-OCR")

process_single_pdf("contract.pdf", model, tokenizer, "out/contract.txt", "out")
text = process_single_image(model, tokenizer, "id_front.jpg", "out")
```

## Design notes

- **Privacy first**: everything runs offline on the local machine. No document content leaves it.
- **Accuracy where it counts**: money tables always go through OCR, because a wrong digit costs more than extra seconds.
- **Cheap checks first**: keyword matching comes before table detection, and table detection comes before OCR. Each step only runs when the cheaper one can't decide.
- **Bounded GPU memory**: the model loads once and is freed explicitly with `torch.cuda.empty_cache()`, so it fits on an 8 GB consumer GPU.
- **Clean per-page state**: the OCR model writes to a fixed `result.md`. Each page reads and deletes it right away so pages never leak into each other.

## Roadmap

- [ ] Command-line arguments instead of hard-coded paths
- [ ] Complete `requirements.txt` (`torch`, `transformers`)
- [ ] Built-in batch entry point for folders of PDFs and images
- [ ] Configurable keyword list and text-length threshold
- [ ] Structured output (JSON) for downstream field extraction
