import os
import torch
from transformers import AutoModel, AutoTokenizer

def run_baidu_unlimited_ocr(image_path, model_dir):
    """
    使用本地部署的 Baidu Unlimited-OCR (3B) 模型识别图片
    """
    print(f"\n=========================================")
    print(f"🚀 Starting Baidu Unlimited-OCR [3B Model]")
    print(f"📂 Image: {image_path}")
    print(f"=========================================")

    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at '{image_path}'")
        return
    if not os.path.exists(model_dir):
        print(f"❌ Error: Model directory not found at '{model_dir}'")
        return

    try:
        print("⏳ Loading Tokenizer & Model into GPU memory...")
        
        # 1. 加载 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, 
            trust_remote_code=True
        )

        # 2. 加载 Model (使用官方推荐的 bfloat16)
        model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        
        # 官方代码直接使用了 .cuda() 将模型推入当前默认显卡
        model = model.eval().cuda() 
        print("✅ Model loaded successfully!")

        print("👁️ Analyzing image content... (Running official 'gundam' mode)")
        
        # 3. 使用官方指定的专属 infer 接口和 prompt 进行单图推理
        # 输出结果默认会保存在当前目录下的 ocr_output 文件夹中
        output_dir = './ocr_output'
        os.makedirs(output_dir, exist_ok=True)
        
        result = model.infer(
            tokenizer,
            prompt='<image>document parsing.',
            image_file=image_path,
            output_path=output_dir,
            base_size=1024, 
            image_size=640, 
            crop_mode=True,
            max_length=32768,
            no_repeat_ngram_size=35, 
            ngram_window=128,
            save_results=True,
        )

        # 4. 打印结果
        print("\n🎉 [OCR RESULT SUCCESS]")
        print("-----------------------------------------")
        # 如果 infer 直接返回了文本，这里会打印；如果不返回，提醒去输出文件夹查看
        if result:
            print(result)
        else:
            print(f"✅ 识别完成！请前往 {output_dir} 文件夹查看保存的结果文件。")
        print("-----------------------------------------")

    except Exception as e:
        print(f"❌ Execution Error: {e}")

    finally:
        # 5. 显存保卫战
        print("\n🧹 Cleaning up GPU memory...")
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        print("✅ GPU Memory released.")

if __name__ == "__main__":
    # 配置区域
    MODEL_PATH = r"D:\models\Unlimited-OCR" 
    IMAGE_PATH = "sample.jpg" 
    
    run_baidu_unlimited_ocr(image_path=IMAGE_PATH, model_dir=MODEL_PATH)