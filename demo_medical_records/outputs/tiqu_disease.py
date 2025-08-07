import os
import time
import json
from zai import ZhipuAiClient

# === GLM4.5 配置 ===
api_key = os.getenv("ZHIPU_API_KEY")
if not api_key:
    raise RuntimeError("请先设置环境变量 ZHIPU_API_KEY 为您的 API Key")
client = ZhipuAiClient(api_key=api_key)

# === 文件路径 ===
input_file = "/home/bmm-system/data/private/yangjianxin/data/demo_medical_records/outputs/full_data_deduplicated.jsonl"
disease_output_file = "translated_diseases.jsonl"
symptom_output_file = "translated_symptoms.jsonl"
batch_size = 10

# === 读取知识图谱数据 ===
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

# === 提取 Disease 和 Symptom 文本 ===
disease_texts = []
symptom_texts = []

# 记录 symptom ID（仅限 presents_as 中用到的）
valid_symptom_ids = set()
disease_id_to_name = {}

# 提取 Disease 实体
for entry in data:
    if entry["type"] == "Disease":
        disease_texts.append(entry["text"])
        disease_id_to_name[entry["id"]] = entry["text"]

# 提取 presents_as 关系，收集与疾病相关的症状 ID
for entry in data:
    if entry["type"] == "relation" and entry["relation"] == "presents_as":
        if entry["head"] in disease_id_to_name:
            valid_symptom_ids.add(entry["tail"])

# 提取 Symptom 实体
for entry in data:
    if entry["type"] == "Symptom" and entry["id"] in valid_symptom_ids:
        symptom_texts.append(entry["text"])

# === Prompt 构建函数（中文）===
def build_prompt(batch, entity_type="症状"):
    numbered_list = "\n".join([f"{i+1}. {text}" for i, text in enumerate(batch)])
    return (
        f"请将以下兽医{entity_type}名称从英文翻译成中文。"
        f"请严格按照以下格式翻译，一定不要添加任何解释或额外内容：\n\n"
        f"1. 英文{entity_type} → 中文翻译\n"
        f"2. 英文{entity_type} → 中文翻译\n"
        f"...\n\n"
        f"{numbered_list}"
    )

# === 翻译函数 ===
def translate_and_save(texts, entity_type, output_file):
    if not texts:
        print(f"⚠️ 未提取到任何 {entity_type} 文本，跳过翻译。")
        return

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            prompt = build_prompt(batch, entity_type)
            print(f"\n翻译 {entity_type} 第 {i} ~ {i+len(batch)-1} 条，Prompt 构造如下：\n{prompt}\n")

            full_output = ""

            try:
                response = client.chat.completions.create(
                    model="glm-4-plus",
                    messages=[{"role": "user", "content": prompt}],
                    thinking={"type": "enabled"},
                    stream=True,
                    max_tokens=800,
                    temperature=0.0,
                )

                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        print(delta.content, end="", flush=True)
                        full_output += delta.content

                print("\n流式输出结束，解析中…")

                for line in full_output.splitlines():
                    if "→" in line:
                        eng, zh = line.split("→", 1)
                        eng = eng.strip().lstrip("0123456789. ").strip()
                        zh = zh.strip()
                        if eng and zh:
                            out_f.write(json.dumps({"english": eng, "chinese": zh}, ensure_ascii=False) + "\n")

                time.sleep(1.5)

            except Exception as e:
                print(f"\n❌ 翻译失败：{e}")
                break

# === 执行翻译 ===
translate_and_save(disease_texts, "疾病", disease_output_file)
translate_and_save(symptom_texts, "症状", symptom_output_file)

print("\n✅ 所有翻译任务完成。")
