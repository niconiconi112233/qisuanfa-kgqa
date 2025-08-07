import json
import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

# ========== 配置 ==========
disease_mention_file = "/home/bmm-system/data/private/yangjianxin/data/demo_medical_records/outputs/translated_diseases.jsonl"
symptom_mention_file = "/home/bmm-system/data/private/yangjianxin/data/demo_medical_records/outputs/translated_symptoms.jsonl"
disease_kb_file = "/home/bmm-system/data/private/yangjianxin/data/demo_medical_records/outputs/unique_diseases.jsonl"
symptom_kb_file = "/home/bmm-system/data/private/yangjianxin/data/demo_medical_records/outputs/unique_symptoms.jsonl"
disease_output_file = "linked_diseases_top5.jsonl"
symptom_output_file = "linked_symptoms_top5.jsonl"
top_k = 5
similarity_threshold = 0.4

# ========== 加载数据 ==========
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

# ========== 主函数 ==========
if __name__ == '__main__':
    print("🔍 Loading BGE-M3 model...")
    model = BGEM3FlagModel("/home/bmm-system/data/private/yangjianxin/data/embedding", use_fp16=True)

    # 加载疾病和症状数据
    print("📄 Loading disease and symptom data...")
    disease_mentions = load_jsonl(disease_mention_file)
    symptom_mentions = load_jsonl(symptom_mention_file)
    disease_kb = load_jsonl(disease_kb_file)
    symptom_kb = load_jsonl(symptom_kb_file)

    # 提取疾病和症状文本
    disease_mention_texts = [m["chinese"] for m in disease_mentions]
    symptom_mention_texts = [m["chinese"] for m in symptom_mentions]
    disease_kb_texts = [e["disease"] for e in disease_kb]
    symptom_kb_texts = [e["symptom"] for e in symptom_kb]

    # 打印加载的数量
    print(f"加载了 {len(disease_mention_texts)} 条疾病文本和 {len(symptom_mention_texts)} 条症状文本。")
    print(f"知识库中有 {len(disease_kb_texts)} 个疾病实体和 {len(symptom_kb_texts)} 个症状实体。")

    # 编码疾病和症状文本
    print("📦 Encoding texts with BGE-M3...")
    disease_mention_embs = model.encode(disease_mention_texts, batch_size=32, return_dense=True)["dense_vecs"]
    symptom_mention_embs = model.encode(symptom_mention_texts, batch_size=32, return_dense=True)["dense_vecs"]
    disease_kb_embs = model.encode(disease_kb_texts, batch_size=32, return_dense=True)["dense_vecs"]
    symptom_kb_embs = model.encode(symptom_kb_texts, batch_size=32, return_dense=True)["dense_vecs"]

    # L2 normalize（保持余弦相似度计算）
    disease_mention_embs = disease_mention_embs / np.linalg.norm(disease_mention_embs, axis=1, keepdims=True)
    symptom_mention_embs = symptom_mention_embs / np.linalg.norm(symptom_mention_embs, axis=1, keepdims=True)
    disease_kb_embs = disease_kb_embs / np.linalg.norm(disease_kb_embs, axis=1, keepdims=True)
    symptom_kb_embs = symptom_kb_embs / np.linalg.norm(symptom_kb_embs, axis=1, keepdims=True)

    # ========= 疾病实体链接 =========
    print("🔗 Performing disease entity linking (Top-5)...")
    linked_diseases = []
    for i, mention_vec in tqdm(enumerate(disease_mention_embs), total=len(disease_mention_embs)):
        sims = np.dot(disease_kb_embs, mention_vec)
        top_indices = sims.argsort()[::-1][:top_k]

        candidates = []
        for idx in top_indices:
            sim_score = float(sims[idx])
            if sim_score >= similarity_threshold:
                candidates.append({
                    "entity": disease_kb_texts[idx],
                    "similarity": round(sim_score, 4)
                })

        linked_diseases.append({
            "mention": disease_mention_texts[i],
            "top_k_candidates": candidates
        })

    # 保存疾病实体链接结果
    print("💾 Saving linked disease results...")
    with open(disease_output_file, "w", encoding="utf-8") as f:
        for r in linked_diseases:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ========= 症状实体链接 =========
    print("🔗 Performing symptom entity linking (Top-5)...")
    linked_symptoms = []
    for i, mention_vec in tqdm(enumerate(symptom_mention_embs), total=len(symptom_mention_embs)):
        sims = np.dot(symptom_kb_embs, mention_vec)
        top_indices = sims.argsort()[::-1][:top_k]

        candidates = []
        for idx in top_indices:
            sim_score = float(sims[idx])
            if sim_score >= similarity_threshold:
                candidates.append({
                    "entity": symptom_kb_texts[idx],
                    "similarity": round(sim_score, 4)
                })

        linked_symptoms.append({
            "mention": symptom_mention_texts[i],
            "top_k_candidates": candidates
        })

    # 保存症状实体链接结果
    print("💾 Saving linked symptom results...")
    with open(symptom_output_file, "w", encoding="utf-8") as f:
        for r in linked_symptoms:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Entity linking (Top-{top_k}) done. Saved to: {disease_output_file} and {symptom_output_file}")
