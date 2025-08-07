import os
import json
import glob

# === 配置区域 ===
input_dir = "/home/bmm-system/data/private/yangjianxin/data/demo_medical_records"   # 存放所有含有 RECORDS 的 .json / .jsonl 文件的目录
unique_diseases_file = "unique_diseases.jsonl"     # 输出去重后疾病名称的 JSONL 文件
unique_symptoms_file = "unique_symptoms.jsonl"     # 输出去重后症状名称的 JSONL 文件

# === 主逻辑 ===
seen_diseases = set()  # 用于去重（存放小写后的疾病名）
seen_symptoms = set()  # 用于去重（存放小写后的症状名）

# 打开输出文件
with open(unique_diseases_file, "w", encoding="utf-8") as out_disease_f, \
     open(unique_symptoms_file, "w", encoding="utf-8") as out_symptom_f:
    
    # 支持 .json 和 .jsonl 后缀
    for filepath in glob.glob(os.path.join(input_dir, "*.[jJ][sS][oO][nNlL]*")):
        # 加载数据
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # 如果是 .jsonl，每行一个 JSON 对象
                f.seek(0)
                data = {"RECORDS": [json.loads(line) for line in f if line.strip()]}

        # 遍历 RECORDS
        for rec in data.get("RECORDS", []):
            # 处理 final_disease 字段
            for disease in rec.get("final_disease", []):
                name = disease.strip()
                key = name.lower()
                if key not in seen_diseases:
                    seen_diseases.add(key)
                    # 写入一行 JSONL
                    out_disease_f.write(json.dumps({"disease": name}, ensure_ascii=False) + "\n")
            
            # 处理 final_symptom 字段
            for symptom in rec.get("final_symptom", []):
                symptom_name = symptom.strip()
                symptom_key = symptom_name.lower()
                if symptom_key not in seen_symptoms:
                    seen_symptoms.add(symptom_key)
                    # 写入一行 JSONL
                    out_symptom_f.write(json.dumps({"symptom": symptom_name}, ensure_ascii=False) + "\n")

print(f"✅ 共写入 {len(seen_diseases)} 个去重后的疾病名称到 {unique_diseases_file}")
print(f"✅ 共写入 {len(seen_symptoms)} 个去重后的症状名称到 {unique_symptoms_file}")
