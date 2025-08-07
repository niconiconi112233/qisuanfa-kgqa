import json
from collections import defaultdict

# ==== 配置路径 ====
translated_disease_file = "translated_diseases.jsonl"
translated_symptom_file = "translated_symptoms.jsonl"
linked_disease_file = "linked_diseases_top5.jsonl"
linked_symptom_file = "linked_symptoms_top5.jsonl"
probability_file = "/home/bmm-system/data/private/yangjianxin/data/demo_medical_records/diseases_symptoms_probability.json"
output_file = "mention_based_probabilities.json"
unmatched_output_file = "unmatched_mentions.json"

SIMILARITY_THRESHOLD = 0.8


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split()) if text else ""


def load_translation_map(path):
    zh2en, en2zh = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            zh = obj["chinese"]
            en = normalize(obj["english"])
            zh2en[zh] = en
            en2zh[en] = zh
    return zh2en, en2zh


def extract_mention_to_entities(path, threshold=SIMILARITY_THRESHOLD):
    mention_to_entities = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            mention = obj["mention"]
            candidates = obj.get("top_k_candidates", [])
            high_sim = [c["entity"] for c in candidates if c["similarity"] >= threshold]
            if high_sim:
                mention_to_entities[mention] = high_sim
            elif candidates:
                mention_to_entities[mention] = [candidates[0]["entity"]]
            else:
                mention_to_entities[mention] = []
    return mention_to_entities


def add(accum, key, value):
    if key in accum:
        accum[key] += value
    else:
        accum[key] = value


# ========== 主流程 ==========
if __name__ == "__main__":
    # 加载翻译文件
    zh2en_disease, en2zh_disease = load_translation_map(translated_disease_file)
    zh2en_symptom, en2zh_symptom = load_translation_map(translated_symptom_file)

    # 加载实体链接映射
    disease_mentions_map = extract_mention_to_entities(linked_disease_file)
    symptom_mentions_map = extract_mention_to_entities(linked_symptom_file)

    # 加载概率表
    with open(probability_file, "r", encoding="utf-8") as f:
        probs = json.load(f)

    # 初始化输出
    output = {
        "disease_prior_prob": {},
        "symptom_prior_prob": {},
        "cond_prob": defaultdict(lambda: defaultdict(float)),
        "joint_prob": defaultdict(lambda: defaultdict(float))
    }

    unmatched_mentions = {
        "unmatched_diseases": [],
        "unmatched_symptoms": []
    }

    # 疾病先验概率
    for mention_zh, linked_zh in disease_mentions_map.items():
        mention_en = zh2en_disease.get(mention_zh)
        if not mention_en:
            unmatched_mentions["unmatched_diseases"].append(mention_zh)
            continue

        total_prob = 0.0
        for zh in linked_zh:
            total_prob += probs.get("diseases_probability", {}).get(zh, 0.0)

        if total_prob > 0:
            output["disease_prior_prob"][mention_en] = total_prob
        else:
            unmatched_mentions["unmatched_diseases"].append(mention_zh)

    # 症状先验概率
    for mention_zh, linked_zh in symptom_mentions_map.items():
        mention_en = zh2en_symptom.get(mention_zh)
        if not mention_en:
            unmatched_mentions["unmatched_symptoms"].append(mention_zh)
            continue

        total_prob = 0.0
        for zh in linked_zh:
            total_prob += probs.get("symptoms_probability", {}).get(zh, 0.0)

        if total_prob > 0:
            output["symptom_prior_prob"][mention_en] = total_prob
        else:
            unmatched_mentions["unmatched_symptoms"].append(mention_zh)

    # 条件概率 P(symptom | disease)
    for dis_zh, dis_linked in disease_mentions_map.items():
        dis_en = zh2en_disease.get(dis_zh)
        if not dis_en:
            continue
        for sym_zh, sym_linked in symptom_mentions_map.items():
            sym_en = zh2en_symptom.get(sym_zh)
            if not sym_en:
                continue
            prob_sum = 0.0
            for zh_d in dis_linked:
                if zh_d in probs.get("symptom_probability_given_disease", {}):
                    cond_probs = probs["symptom_probability_given_disease"][zh_d]
                    for zh_s in sym_linked:
                        prob_sum += cond_probs.get(zh_s, 0.0)
            if prob_sum > 0:
                output["cond_prob"][dis_en][sym_en] += prob_sum

    # 联合概率 P(sym1, sym2)
    for zh1, linked1 in symptom_mentions_map.items():
        en1 = zh2en_symptom.get(zh1)
        if not en1:
            continue
        for zh2, linked2 in symptom_mentions_map.items():
            en2 = zh2en_symptom.get(zh2)
            if not en2 or en1 >= en2:
                continue  # 保证对称 & 去重
            prob_sum = 0.0
            for s1 in linked1:
                for s2 in linked2:
                    p = probs.get("symptoms_joint_probability", {}).get(f"{s1}|||{s2}", 0.0)
                    if p == 0.0:
                        p = probs.get("symptoms_joint_probability", {}).get(f"{s2}|||{s1}", 0.0)
                    prob_sum += p
            if prob_sum > 0:
                output["joint_prob"][en1][en2] += prob_sum
                output["joint_prob"][en2][en1] += prob_sum  # 对称回写

    # 保存输出
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    with open(unmatched_output_file, "w", encoding="utf-8") as f:
        json.dump(unmatched_mentions, f, ensure_ascii=False, indent=2)

    print(f"✅ mention 概率表已保存至：{output_file}")
    print(f"📄 未匹配 mention 记录已保存至：{unmatched_output_file}")
