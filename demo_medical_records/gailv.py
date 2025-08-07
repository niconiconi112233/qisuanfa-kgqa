#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 demo*.json 生成四张概率表
-------------------------------------------
output:
    diseases_symptoms_probability.json
        {
          "diseases_probability": {...},
          "symptoms_probability": {...},
          "symptom_probability_given_disease": {...},
          "symptoms_joint_probability": {...}
        }
"""

import os, json, unicodedata, argparse
from collections import Counter, defaultdict
from itertools import combinations
from tqdm import tqdm

# ---------- 参数 ----------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/bmm-system/data/private/yangjianxin/data/demo_medical_records", help="包含 demo*.json 的文件夹")
    ap.add_argument("--pattern", default="demo", help="文件名前缀 demo")
    ap.add_argument("--out", default="diseases_symptoms_probability.json")
    return ap.parse_args()


# ---------- 简易规范化 ----------
def canon(text: str) -> str:
    if not text: return ""
    t = unicodedata.normalize("NFKC", text.strip())
    # 统一中文括号 → ()
    t = t.replace("（", "(").replace("）", ")")
    return t


# ---------- 主逻辑 ----------
def main():
    args = get_args()

    # 计数器
    disease_cnt             = Counter()
    symptom_cnt             = Counter()
    disease_symptom_cnt     = defaultdict(Counter)
    symptom_pair_cnt        = Counter()

    seen_records = set()
    total_cases  = 0

    # 遍历文件
    files = sorted(os.path.join(args.dir, f)
                   for f in os.listdir(args.dir)
                   if f.startswith(args.pattern) and f.endswith(".json"))

    for fp in files:
        data = json.load(open(fp, encoding="utf-8"))
        for rec in tqdm(data.get("RECORDS", []), desc=os.path.basename(fp)):
            rid = rec.get("record_id") or rec.get("id")
            if rid in seen_records:
                continue
            seen_records.add(rid)

            diseases = { canon(d) for d in rec.get("final_disease", []) if canon(d) }
            symptoms = { canon(s) for s in rec.get("final_symptom", []) if canon(s) }

            if not diseases and not symptoms:
                continue

            total_cases += 1
            disease_cnt.update(diseases)
            symptom_cnt.update(symptoms)

            for d in diseases:
                disease_symptom_cnt[d].update(symptoms)

            for s1, s2 in combinations(sorted(symptoms), 2):
                symptom_pair_cnt[(s1, s2)] += 1

    print("有效病例数 N =", total_cases)

    # ---------- 概率 ----------
    P_disease  = {d: c / total_cases for d, c in disease_cnt.items()}
    P_symptom  = {s: c / total_cases for s, c in symptom_cnt.items()}
    P_s_given_d = {
        d: {s: cs / disease_cnt[d] for s, cs in cnt.items()}
        for d, cnt in disease_symptom_cnt.items()
    }
    P_joint = {f"{s1}|||{s2}": c / total_cases
               for (s1, s2), c in symptom_pair_cnt.items()}

    out = {
        "diseases_probability": P_disease,
        "symptoms_probability": P_symptom,
        "symptom_probability_given_disease": P_s_given_d,
        "symptoms_joint_probability": P_joint,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("已保存 →", args.out)


if __name__ == "__main__":
    main()
