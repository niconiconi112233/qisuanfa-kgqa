import os

def remove_matched_lines(md_folder_path, txt_file_path):
    # 获取所有 .md 文件的文件名（去除扩展名）
    md_filenames = {
        os.path.splitext(f)[0]
        for f in os.listdir(md_folder_path)
        if f.endswith('.md')
    }

    # 读取 .txt 文件的所有行
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 去除那些文件名在 md 文件中出现的行
    filtered_lines = [line for line in lines if line.strip() not in md_filenames]

    # 将保留的行写回文件
    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)

    print(f"处理完成，已删除 {len(lines) - len(filtered_lines)} 行。")

# 使用示例（请替换为你的真实路径）
md_folder = '/Users/yangjianxin/Downloads/open_deep_research-main/src'
txt_file = '/Users/yangjianxin/Downloads/open_deep_research-main/src/matched_diseases.txt'

remove_matched_lines(md_folder, txt_file)
