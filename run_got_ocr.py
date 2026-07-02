import os
import torch
from transformers import AutoModel, AutoTokenizer

def run_local_got_ocr(image_path, model_dir, mode="plain"):
    """
    使用本地部署的 GOT-OCR 2.0 模型识别图片
    :param image_path: 图片的本地路径
    :param model_dir: GOT-OCR2_0 模型的本地绝对路径
    :param mode: "plain" 代表识别纯文本（适合身份证/截图）；"format" 代表识别格式化内容（适合复杂表格/公式）
    """
    print(f"\n=========================================")
    print(f"🚀 Starting GOT-OCR 2.0 [Mode: {mode}]")
    print(f"📂 Image: {image_path}")
    print(f"=========================================")

    # 1. 严格检查路径
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at '{image_path}'")
        return None
    if not os.path.exists(model_dir):
        print(f"❌ Error: Model directory not found at '{model_dir}'")
        return None

    # 2. 映射推理模式参数
    # GOT-OCR 2.0 原生支持: 'ocr' (纯文本), 'format' (格式化如Markdown表格)
    got_ocr_type = "ocr" if mode == "plain" else "format"

    try:
        # 3. 加载模型至 RTX 5070 (8GB 显存优化配置)
        print("⏳ Loading model into GPU memory (Takes around 5-10s)...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,     # 优化内存占用
            device_map="cuda",          # 强制指定使用 GPU 推理
            use_safetensors=True,       # 安全且快速地读取权重
            pad_token_id=tokenizer.eos_token_id
        )
        model = model.eval().cuda()     # 切换为评估模式
        print("✅ Model loaded successfully!")

        # 4. 执行图片识别
        print("👁️ Analyzing image content...")
        with torch.no_grad():           # 关闭梯度计算，进一步节省显存
            result = model.chat(tokenizer, image_path, ocr_type=got_ocr_type)
        
        print("\n🎉 [OCR RESULT SUCCESS]")
        print("-----------------------------------------")
        print(result)
        print("-----------------------------------------")
        
        return result

    except Exception as e:
        print(f"❌ Execution Error: {e}")
        return None

    finally:
        # 5. 显存清理防爆（针对 8GB 显存的硬约束）
        print("🧹 Cleaning up GPU memory...")
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        print("✅ GPU Memory released.")

if __name__ == "__main__":
    # 【配置区域】请根据你的实际情况修改这三个变量
    
    # 1. 模型的绝对路径（就是你刚刚下载好的地方）
    MODEL_PATH = r"D:\models\GOT-OCR2_0"
    
    # 2. 你要识别的图片路径（确保这个图片和脚本在同一个文件夹，或者写绝对路径）
    IMAGE_PATH = "sample.jpg" 
    
    # 3. 运行模式选择：
    # 如果是识别身份证，用 "plain"
    # 如果是识别之前那个换行的表格，用 "format"
    RUN_MODE = "plain" 
    
    # 执行
    run_local_got_ocr(image_path=IMAGE_PATH, model_dir=MODEL_PATH, mode=RUN_MODE)