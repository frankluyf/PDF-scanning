import os
import torch
import fitz
import pdfplumber

from run_baidu_ocr_pdf import load_baidu_ocr_model, process_page_with_ocr
from run_pdfplumber import extract_text_with_plumber
from clean import clean_output_dir

# 出现这些词，说明本页大概率涉及金额/凭证，表格必须抽准 → 倾向走 OCR
AMOUNT_KEYWORDS = (
    "金额", "大写", "人民币", "¥", "￥", "元整", "圆整", "本金",
    "放款", "代偿", "偿还", "还款", "利息", "币种", "收款",
)


def should_ocr_for_amount(page_fitz, probe_text):
    """
    稳健判断：这一页是不是‘含金额的表格页’，是就交给 OCR。
    先看关键词（便宜），命中了再花代价去 find_tables（较慢）。
    没关键词的纯条款页直接返回 False，走 pdfplumber。
    """
    if not any(kw in probe_text for kw in AMOUNT_KEYWORDS):
        return False
    try:
        has_tables = len(page_fitz.find_tables().tables) > 0
    except Exception:
        has_tables = False
    return has_tables
# ──────────────────────────────────────────────────────────────

def process_single_pdf(pdf_path, model, tokenizer, output_txt_path, ocr_work_dir):
    """
    用【已加载好的】model/tokenizer 处理单个 PDF，结果写入 output_txt_path。
    注意：这里不再加载/卸载模型、也不清理目录——那些交给调用方统一管理，
    这样批处理时模型只加载一次，效率高很多。
    返回 True 表示成功。
    """
    print(f"\n{'='*41}")
    print(f"📂 处理文件: {pdf_path}")
    print(f"{'='*41}")

    if not os.path.exists(pdf_path):
        print("❌ File not found!")
        return False

    doc_fitz = None
    doc_plumber = None
    try:
        doc_fitz = fitz.open(pdf_path)
        doc_plumber = pdfplumber.open(pdf_path)

        with open(output_txt_path, "w", encoding="utf-8") as final_file:
            for i in range(len(doc_fitz)):
                page_num = i + 1
                page_fitz = doc_fitz[i]

                probe_text = page_fitz.get_text()
                is_low_text = len(probe_text.strip()) < 10

                final_file.write(f"\n\n{'='*20} Page {page_num} {'='*20}\n\n")

                page_content = ""

                # ① 文字太少 → 扫描件/图片页，只能 OCR
                if is_low_text:
                    print(f"[Page {page_num}] ⚠️ 文字稀少，判为扫描/图片页 → OCR")
                    page_content = process_page_with_ocr(
                        model, tokenizer, page_fitz, page_num, ocr_work_dir)

                # ② 含金额且有表格结构 → 走 OCR 保证金额表不被抽错
                elif should_ocr_for_amount(page_fitz, probe_text):
                    print(f"[Page {page_num}] 💰 金额表格页 → OCR（保精度）")
                    page_content = process_page_with_ocr(
                        model, tokenizer, page_fitz, page_num, ocr_work_dir)

                # ③ 其余纯文本页 → 信任 pdfplumber，抽不到再兜底 OCR
                else:
                    print(f"[Page {page_num}] 📝 纯文本页，pdfplumber 快速抽取")
                    page_plumber = doc_plumber.pages[i]
                    page_content = extract_text_with_plumber(page_plumber)

                    if not page_content or not page_content.strip():
                        print(f"   -> 🚨 兜底：pdfplumber 没抽到，转 OCR 重扫第 {page_num} 页")
                        page_content = process_page_with_ocr(
                            model, tokenizer, page_fitz, page_num, ocr_work_dir)

                final_file.write(page_content if page_content else "[此页无提取内容]")

        print(f"✅ 已保存: {output_txt_path}")
        return True

    except Exception as e:
        print(f"❌ Processing error: {e}")
        return False
    finally:
        if doc_fitz is not None:
            doc_fitz.close()
        if doc_plumber is not None:
            doc_plumber.close()


def main(pdf_path, model_dir, output_dir="output_text"):
    """单文件入口：保持原来的行为（加载模型 → 清目录 → 处理 → 释放显存）。"""
    clean_output_dir(output_dir)
    final_output_path = os.path.join(output_dir, "final_output.txt")

    model, tokenizer = load_baidu_ocr_model(model_dir)
    try:
        process_single_pdf(pdf_path, model, tokenizer, final_output_path, output_dir)
    finally:
        print("\n🧹 Cleaning up GPU memory...")
        del model, tokenizer
        torch.cuda.empty_cache()
        print("✅ GPU Memory released. Process Finished.")


if __name__ == "__main__":
    MODEL_PATH = r"D:\models\Unlimited-OCR"
    TEST_PDF = r".\input\sample2.pdf"
    main(pdf_path=TEST_PDF, model_dir=MODEL_PATH)