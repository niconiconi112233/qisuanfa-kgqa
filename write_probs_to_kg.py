#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据来源（顶层 keys）：
- disease_prior          -> 疾病实体["概率"].p_d / count_d
- symptom_prior          -> 症状实体["概率"].p_s / count_s
- cond_p_s_given_d       -> 已存在的 疾病→症状 关系["权重"].p_s_given_d（不新建）
- symptom_joint          -> 已存在的 症状→症状 关系["权重"].p_joint（不新建，默认也不做双向）
把 probabilities.json 的概率写入图谱：
- 允许创建新边，但仅在两端节点（疾病/症状）都已存在于图谱时才创建。
- 未匹配实体一律跳过，不写、不建。
- 避免重复建边（按 (src, tgt, rel_type) 去重）。

输入：
- disease_prior
- symptom_prior
- cond_p_s_given_d
- symptom_joint
"""

import json
import argparse
from copy import deepcopy
from typing import Dict, List, Tuple, Any

# 自动探测常见关系字段
SOURCE_NAME_KEYS = ["来源名称", "source_name", "来源", "起点", "head_name", "subject", "from", "源"]
TARGET_NAME_KEYS = ["目标名称", "target_name", "目标", "终点", "tail_name", "object", "to", "汇"]
REL_TYPE_KEYS   = ["关系类型", "predicate", "rel", "关系"]

def norm(s: Any) -> str:
    if s is None: return ""
    return str(s).strip().replace("\u3000", " ")

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dict(obj: dict, key: str) -> dict:
    val = obj.get(key)
    if not isinstance(val, dict):
        val = {}
        obj[key] = val
    return val

def detect_rel_keys(relations: List[dict]) -> Tuple[str, str, str]:
    if not relations:
        return "", "", ""
    sample = relations[0]
    src_key = next((k for k in SOURCE_NAME_KEYS if k in sample), "")
    tgt_key = next((k for k in TARGET_NAME_KEYS if k in sample), "")
    rel_key = next((k for k in REL_TYPE_KEYS   if k in sample), "")
    return src_key, tgt_key, rel_key

def index_entities(entities: List[dict]) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    dis, sym = {}, {}
    for e in entities:
        t = norm(e.get("类型", ""))
        n = norm(e.get("名称", ""))
        if not n: continue
        if t == "疾病":   dis[n] = e
        if t == "症状":   sym[n] = e
    return dis, sym

def build_rel_index(relations: List[dict], src_key: str, tgt_key: str, rel_key: str) -> Dict[Tuple[str,str,str], dict]:
    idx = {}
    for r in relations:
        s = norm(r.get(src_key, "")) if src_key else ""
        t = norm(r.get(tgt_key, "")) if tgt_key else ""
        rt = norm(r.get(rel_key, "")) if rel_key else ""
        idx[(s, t, rt)] = r
    return idx

def get_or_create_relation(relations: List[dict], rel_index: Dict[Tuple[str,str,str], dict],
                           src_key: str, tgt_key: str, rel_key: str,
                           s: str, t: str, rtype: str) -> Tuple[dict, bool]:
    """存在则返回；否则创建一条（并登记到索引）。"""
    k = (s if src_key else "", t if tgt_key else "", rtype if rel_key else "")
    r = rel_index.get(k)
    if r:
        return r, False
    r = {}
    if src_key: r[src_key] = s
    if tgt_key: r[tgt_key] = t
    if rel_key: r[rel_key] = rtype
    relations.append(r)
    rel_index[k] = r
    return r, True

def main():
    ap = argparse.ArgumentParser(description="写入概率，允许加边但仅限两端实体存在于图谱。")
    ap.add_argument("--kg_in", required=True)
    ap.add_argument("--kg_out", required=True)
    ap.add_argument("--all_in_one", required=True, help="probabilities.json 路径")
    ap.add_argument("--cond_rel_type", default="疾病-症状")
    ap.add_argument("--joint_rel_type", default="症状共现")
    ap.add_argument("--joint_bidirectional", action="store_true", help="共现是否双向建边")
    args = ap.parse_args()

    kg = load_json(args.kg_in)
    kg = deepcopy(kg)
    entities: List[dict] = kg.get("实体", [])
    relations: List[dict] = kg.get("关系", [])
    if not isinstance(relations, list):
        relations = []
        kg["关系"] = relations

    src_key, tgt_key, rel_key = detect_rel_keys(relations)
    print(f"[INFO] 关系字段：src='{src_key}', tgt='{tgt_key}', rel='{rel_key}'")

    dis_map, sym_map = index_entities(entities)
    print(f"[INFO] 疾病实体数：{len(dis_map)}；症状实体数：{len(sym_map)}")

    data = load_json(args.all_in_one)
    disease_prior = data.get("disease_prior", []) or []
    symptom_prior = data.get("symptom_prior", []) or []
    cond_list     = data.get("cond_p_s_given_d", []) or []
    joint_list    = data.get("symptom_joint", []) or []

    rel_index = build_rel_index(relations, src_key, tgt_key, rel_key)

    stat = {
        "disease_prior_total": len(disease_prior),
        "disease_prior_written": 0,
        "disease_prior_unmatched": 0,

        "symptom_prior_total": len(symptom_prior),
        "symptom_prior_written": 0,
        "symptom_prior_unmatched": 0,

        "cond_total": len(cond_list),
        "cond_edge_updated": 0,
        "cond_edge_created": 0,
        "cond_unmatched_disease": 0,
        "cond_unmatched_symptom": 0,

        "joint_total": len(joint_list),
        "joint_edge_updated": 0,
        "joint_edge_created": 0,
        "joint_unmatched_symptom_i": 0,
        "joint_unmatched_symptom_j": 0,
    }

    # 实体概率
    for row in disease_prior:
        d = norm(row.get("disease")); p = row.get("p_d", None)
        if not d or p is None: continue
        ent = dis_map.get(d)
        if not ent:
            stat["disease_prior_unmatched"] += 1; continue
        prob = ensure_dict(ent, "概率")
        prob["p_d"] = float(p)
        if "count" in row: prob["count_d"] = int(row["count"])
        stat["disease_prior_written"] += 1

    for row in symptom_prior:
        s = norm(row.get("symptom")); p = row.get("p_s", None)
        if not s or p is None: continue
        ent = sym_map.get(s)
        if not ent:
            stat["symptom_prior_unmatched"] += 1; continue
        prob = ensure_dict(ent, "概率")
        prob["p_s"] = float(p)
        if "count" in row: prob["count_s"] = int(row["count"])
        stat["symptom_prior_written"] += 1

    # 条件概率：仅当疾病和症状实体都存在时才更新/创建
    for row in cond_list:
        d = norm(row.get("disease")); s = norm(row.get("symptom"))
        p = row.get("p_s_given_d", None)
        if not d or not s or p is None: continue
        if d not in dis_map:
            stat["cond_unmatched_disease"] += 1; continue
        if s not in sym_map:
            stat["cond_unmatched_symptom"] += 1; continue

        rel_obj, created = get_or_create_relation(
            relations, rel_index, src_key, tgt_key, rel_key,
            s=d, t=s, rtype=args.cond_rel_type
        )
        w = ensure_dict(rel_obj, "权重")
        w["p_s_given_d"] = float(p)
        if "num" in row: w["num"] = int(row["num"])
        if "den" in row: w["den"] = int(row["den"])
        if created: stat["cond_edge_created"] += 1
        else:       stat["cond_edge_updated"] += 1

    # 共现：仅当两端症状实体都存在时才更新/创建
    def upsert_joint(a: str, b: str, row: dict):
        rel_obj, created = get_or_create_relation(
            relations, rel_index, src_key, tgt_key, rel_key,
            s=a, t=b, rtype=args.joint_rel_type
        )
        w = ensure_dict(rel_obj, "权重")
        w["p_joint"] = float(row["p_joint"])
        if "count" in row: w["count"] = int(row["count"])
        if "den" in row:   w["den"]   = int(row["den"])
        return created

    for row in joint_list:
        si = norm(row.get("symptom_i")); sj = norm(row.get("symptom_j"))
        pj = row.get("p_joint", None)
        if not si: stat["joint_unmatched_symptom_i"] += 1; continue
        if not sj: stat["joint_unmatched_symptom_j"] += 1; continue
        if pj is None: continue
        if si not in sym_map:
            stat["joint_unmatched_symptom_i"] += 1; continue
        if sj not in sym_map:
            stat["joint_unmatched_symptom_j"] += 1; continue

        if upsert_joint(si, sj, row):
            stat["joint_edge_created"] += 1
        else:
            stat["joint_edge_updated"] += 1

        if args.joint_bidirectional and si != sj:
            if upsert_joint(sj, si, row):
                stat["joint_edge_created"] += 1
            else:
                stat["joint_edge_updated"] += 1

    # 统计
    print("\n=== 写入统计（仅在两端实体存在时才建/改边） ===")
    print(f"疾病先验：总 {stat['disease_prior_total']}  条；写入 {stat['disease_prior_written']}，未匹配 {stat['disease_prior_unmatched']}")
    print(f"症状先验：总 {stat['symptom_prior_total']}  条；写入 {stat['symptom_prior_written']}，未匹配 {stat['symptom_prior_unmatched']}")
    print(f"条件概率：总 {stat['cond_total']}  条；更新边 {stat['cond_edge_updated']}，新建边 {stat['cond_edge_created']}，未匹配疾病 {stat['cond_unmatched_disease']}，未匹配症状 {stat['cond_unmatched_symptom']}")
    print(f"联合概率：总 {stat['joint_total']} 条；更新边 {stat['joint_edge_updated']}，新建边 {stat['joint_edge_created']}，未匹配 symptom_i {stat['joint_unmatched_symptom_i']}，未匹配 symptom_j {stat['joint_unmatched_symptom_j']}")

    with open(args.kg_out, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 已写出：{args.kg_out}")

if __name__ == "__main__":
    main()
