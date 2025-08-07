import json

# === 输入输出文件路径 ===
translated_files = {
    "disease": "translated_diseases.jsonl",
    "symptom": "translated_symptoms.jsonl",
}
linked_files = {
    "disease": "linked_diseases_top5.jsonl",
    "symptom": "linked_symptoms_top5.jsonl",
}
output_files = {
    "disease": "linked_diseases_with_english.jsonl",
    "symptom": "linked_symptoms_with_english.jsonl",
}


def load_translation_map(path):
    """加载翻译文件为 {chinese: english} 字典"""
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            mapping[data['chinese']] = data['english']
    return mapping


def add_english_to_linked(input_path, output_path, zh2en_map):
    """给实体链接结果添加 english 字段"""
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line.strip())
            mention = data['mention']
            english = zh2en_map.get(mention)
            data['english'] = english if english else ""  # 没找到时填空字符串
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")


# === 主流程 ===
for key in ["disease", "symptom"]:
    zh2en_map = load_translation_map(translated_files[key])
    add_english_to_linked(linked_files[key], output_files[key], zh2en_map)

print("✅ 添加完成，结果已保存到带 _with_english 的文件中。")
