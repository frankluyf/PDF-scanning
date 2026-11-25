import os
import torch
import fitz
import shutil  # 🌟 需要引入 shutil 来删除模型生成的文件夹
from transformers import AutoModel, AutoTokenizer

def load_baidu_ocr_model(model_dir):
    print("⏳ Loading Baidu Unlimited-OCR into GPU memory...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    print("✅ Model loaded successfully!\n")
    return model, tokenizer

def process_page_with_ocr(model, tokenizer, page_fitz, page_num, output_dir):
    """渲染图片并交由OCR处理，读取固定的 result.md 后清理现场"""
    temp_img_name = f"temp_page_{page_num}.png"
    temp_img_path = os.path.join(output_dir, temp_img_name)
    
    mat = fitz.Matrix(200 / 72, 200 / 72)
    pix = page_fitz.get_pixmap(matrix=mat)
    pix.save(temp_img_path)
    
    # 让模型执行推理并保存结果
    model.infer(
        tokenizer,
        prompt='<image>document parsing.',
        image_file=temp_img_path,
        output_path=output_dir,
        base_size=1024, 
        image_size=640, 
        crop_mode=True,
        max_length=4096,
        no_repeat_ngram_size=35, 
        ngram_window=128,
        save_results=True,  
    )
    
    # 🌟 破案关键：百度模型硬编码了输出文件名！
    expected_md = os.path.join(output_dir, "result.md")
    expected_box_img = os.path.join(output_dir, "result_with_boxes.jpg")
    expected_img_dir = os.path.join(output_dir, "images")
    
    extracted_content = ""
    
    # 1. 抓取生成的 result.md
    if os.path.exists(expected_md):
        with open(expected_md, "r", encoding="utf-8") as f:
            extracted_content = f.read()
        os.remove(expected_md)  # 读完立刻删除，防止污染下一页
    else:
        print(f"   -> ⚠️ 警告：未找到 result.md。")
    
    # 2. 清理模型附带生成的垃圾文件（画框图和切片文件夹）
    if os.path.exists(expected_box_img):
        os.remove(expected_box_img)
    if os.path.exists(expected_img_dir):
        shutil.rmtree(expected_img_dir)
    
    # 3. 清理我们自己生成的临时 PDF 截图
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
        
    print(f"   -> 🧠 OCR Success! Complex content extracted as Markdown string.")
    
    return extracted_content


# ============================================================
# 把下面这个函数【追加】到 PDF-scanning/run_baidu_ocr_pdf.py 末尾
# 不要改动已有的 load_baidu_ocr_model / process_page_with_ocr
# ============================================================
import os
import glob


def process_single_image(model, tokenizer, image_path, ocr_work_dir):
    """
    用【已加载好的】model/tokenizer 直接识别一张图片，返回识别出的文本。
    图片本身就是"一页"，不需要 fitz/pdfplumber 解析，直接喂 infer。
    参数接口对齐你 run_baidu_unlimited_ocr 里用的 model.infer(...)。
    """
    out_dir = os.path.join(ocr_work_dir, "img_ocr_tmp")
    os.makedirs(out_dir, exist_ok=True)

    with torch.inference_mode():
        result = model.infer(
            tokenizer,
            prompt="<image>document parsing.",
            image_file=image_path,
            output_path=out_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            max_length=4096,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=True,
        )

    # infer 可能直接返回文本，也可能只把结果写进 output_path。两种都兜住：
    if result and isinstance(result, str) and result.strip():
        return result.strip()

    # 兜底：从 output_path 里把最新生成的文本文件读出来
    txt_files = glob.glob(os.path.join(out_dir, "*.txt")) + \
                glob.glob(os.path.join(out_dir, "*.md"))
    if txt_files:
        latest = max(txt_files, key=os.path.getmtime)
        try:
            with open(latest, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"   ⚠️ 读取 OCR 结果文件失败: {e}")

    return ""