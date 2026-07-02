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
    
    mat = fitz.Matrix(300 / 72, 300 / 72)
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
        max_length=32768,
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