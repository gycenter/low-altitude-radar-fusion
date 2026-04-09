import os
import shutil
import re

# 该脚本是将最初的原始回波数据改为雷达处理模块所需的按标签分类的data集 

# 设置原始文件夹路径
root_dir = 'dataset/原始回波'
output_dir = 'dataset/data'

# 遍历原始文件夹中的所有文件
for filename in os.listdir(root_dir):
    if filename.endswith(".dat"):
        # 使用正则提取 label（格式为 *_Label_数字.dat）
        match = re.search(r'_Label_(\d)\.dat$', filename)
        if match:
            label = match.group(1)
            target_dir = os.path.join(output_dir, f"label{label}","原始回波")
            os.makedirs(target_dir, exist_ok=True)  # 若子目录不存在则创建
            src_path = os.path.join(root_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            shutil.copy2(src_path, dst_path)
            # print(f"已移动: {filename} -> {target_dir}")

print("done")
