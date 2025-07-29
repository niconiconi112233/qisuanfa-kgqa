import pandas as pd

# === 1. 读取 Claude 输出的 200 个疾病名 ===
txt_file = "claude_top200_diseases.txt"
with open(txt_file, "r", encoding="utf-8") as f:
    predicted_diseases = [line.strip() for line in f if line.strip()]

# === 2. 读取原始 Excel 中的“英文名全称”列 ===
xlsx_file = "/Users/yangjianxin/Downloads/open_deep_research-main/src/疾病诊断库xlsx.xlsx"
column_name = "英文名全称"

df = pd.read_excel(xlsx_file)
original_terms = df[column_name].dropna().astype(str).tolist()

# 转成小写集合（匹配时忽略大小写）
original_set = set(term.lower() for term in original_terms)

# === 3. 保留匹配成功的疾病名 ===
matched_diseases = [disease for disease in predicted_diseases if disease.lower() in original_set]

# === 4. 保存匹配结果到新文件 ===
output_file = "matched_diseases.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(matched_diseases))

# === 5. 打印统计信息 ===
total = len(predicted_diseases)
matched = len(matched_diseases)
print(f"✅ 匹配成功：{matched}/{total} 个疾病名保留")
print(f"📄 匹配结果已保存到：{output_file}")
