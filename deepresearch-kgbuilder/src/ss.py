import os
import shutil

# 设置源文件夹和目标文件夹路径
source_folder = '/Users/yangjianxin/Downloads/open_deep_research-main/src'
target_folder = '/Users/yangjianxin/Downloads/open_deep_research-main/src/report_0711'

# 如果目标文件夹不存在，则创建
os.makedirs(target_folder, exist_ok=True)

# 遍历源文件夹下的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('.md'):
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)
        # 移动文件
        shutil.move(source_path, target_path)
        print(f'Moved: {filename}')
