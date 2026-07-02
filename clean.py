import os
import shutil

def clean_output_dir(dir_path="output_text"):
    """
    清空指定的目录。如果目录不存在，则创建它。
    """
    print(f"--- 准备清理目录: {dir_path} ---")
    
    # 检查目录是否存在
    if os.path.exists(dir_path):
        try:
            # 遍历目录下的所有文件和文件夹
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                
                # 判断是文件还是文件夹，分别进行删除
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                    print(f"已删除文件: {filename}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path) # 删除子文件夹及其内容
                    print(f"已删除文件夹: {filename}")
                    
            print(f"✅ 目录 '{dir_path}' 已清理完毕。")
            
        except Exception as e:
            print(f"❌ 清理目录时发生错误: {e}")
    else:
        # 如果目录不存在，提示并创建（确保后续程序不会因为找不到目录报错）
        print(f"目录 '{dir_path}' 不存在，正在为您创建...")
        try:
            os.makedirs(dir_path)
            print(f"✅ 已创建空目录: {dir_path}")
        except Exception as e:
            print(f"❌ 创建目录时发生错误: {e}")

# 测试运行
if __name__ == "__main__":
    # 默认清理当前目录下的 output_text 文件夹
    clean_output_dir()