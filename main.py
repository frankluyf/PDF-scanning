import os
import torch
import fitz
import pdfplumber

from run_baidu_ocr_pdf import load_baidu_ocr_model, process_page_with_ocr
from run_pdfplumber import extract_text_with_plumber
from clean import clean_output_dir  

def main(pdf_path, model_dir, output_dir="output_text"):
    print(f"=========================================")
    print(f"🚀 Starting Single-File Hybrid Pipeline")
    print(f"📂 File: {pdf_path}")
    print(f"=========================================")
    
    if not os.path.exists(pdf_path):
        print("❌ File not found!")
        return

    # 清理文件夹
    clean_output_dir(output_dir)
    
    # 设定最终输出的合并文件名
    final_output_path = os.path.join(output_dir, "final_output.txt")
    
    # 1. 预加载模型
    model, tokenizer = load_baidu_ocr_model(model_dir)

    try:
        doc_fitz = fitz.open(pdf_path)
        doc_plumber = pdfplumber.open(pdf_path)
        
        # 🌟 打开最终的 txt 文件，准备写入
        with open(final_output_path, "w", encoding="utf-8") as final_file:
            
            for i in range(len(doc_fitz)):
                page_num = i + 1
                page_fitz = doc_fitz[i]
                
                probe_text = page_fitz.get_text()
                is_low_text = len(probe_text.strip()) < 10
                has_images = len(page_fitz.get_images()) > 0
                has_tables = len(page_fitz.find_tables().tables) > 0
                
                # 写入页码分隔符，让文档结构更清晰
                final_file.write(f"\n\n{'='*20} Page {page_num} {'='*20}\n\n")
                
                page_content = ""
                
                # 4. 分支路由
                if is_low_text or has_images or has_tables:
                    trigger_reason = "Low Text" if is_low_text else ("Tables" if has_tables else "Images")
                    print(f"[Page {page_num}] ⚠️ Complex content ({trigger_reason}). Triggering OCR...")
                    page_content = process_page_with_ocr(model, tokenizer, page_fitz, page_num, output_dir)
                else:
                    print(f"[Page {page_num}] 📝 Text-dense page. Extracting fast...")
                    page_plumber = doc_plumber.pages[i]
                    page_content = extract_text_with_plumber(page_plumber)
                    
                    # 🌟 终极兜底：如果 pdfplumber 失效，强制切回 OCR！
                    if not page_content or not page_content.strip():
                        print(f"   -> 🚨 兜底机制触发：未提取到有效文字！强制使用大模型 OCR 重新扫描第 {page_num} 页...")
                        page_content = process_page_with_ocr(model, tokenizer, page_fitz, page_num, output_dir)
                
                # 将当前页的内容写入最终文件
                if page_content:
                    final_file.write(page_content)
                else:
                    final_file.write("[此页无提取内容]")

        print(f"\n🎉 识别完成！所有内容已成功合并保存至: {final_output_path}")

    except Exception as e:
        print(f"❌ Processing error: {e}")
        
    finally:
        if 'doc_fitz' in locals(): doc_fitz.close()
        if 'doc_plumber' in locals(): doc_plumber.close()
        
        print("\n🧹 Cleaning up GPU memory...")
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        torch.cuda.empty_cache()
        print("✅ GPU Memory released. Process Finished.")

if __name__ == "__main__":
    MODEL_PATH = r"D:\models\Unlimited-OCR" 
    TEST_PDF = ".\input\sample2.pdf" 
    
    main(pdf_path=TEST_PDF, model_dir=MODEL_PATH)