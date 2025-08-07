#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 mention_based_probabilities.json 写入知识图谱 (JSONL)

输入
-----
1) 概率文件：
   {
     "disease_prior_prob": { "actinomycosis": p, ... },
     "symptom_prior_prob": { "acute abdomen": p, ... },
     "cond_prob": { "actinomycosis": {"acute abdomen": p, ...}, ... },
     "joint_prob": { "acute abdomen": {"abdominal pain": p, ...}, ... }
   }


2) 知识图谱 JSONL：
   - Disease 节点: {"type":"Disease","id":"ENT_xxx","text":"Actinomycosis", ...}
   - Symptom 节点: {"type":"Symptom","id":"ENT_xxx","text":"Acute abdomen", ...}
   - Relation: {"type":"relation","head":"ENT_...","relation":"presents_as"/"co_occurs_with","tail":"ENT_...","attr":{...}}

写入
-----
- Disease.attr.prior = P(D)
- Symptom.attr.prior = P(S)
- relation (presents_as).attr.weight = P(S|D)
- relation (co_occurs_with).attr.weight = P(Si,Sj)   # 无向，这里写成对称两条

匹配规则
--------
- 用英文名匹配，忽略大小写与空格（不创建新节点，除非传 --create_nodes）
- 仅疾病匹配 Disease 节点；仅症状匹配 Symptom 节点

输出
-----
- --kg_out: 更新后的 KG
- unmatched_report.json: 未匹配统计与明细
- 终端打印匹配统计
"""

import os
import re
import json
import argparse
from collections import defaultdict

# ---------------------- CLI ----------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_path", required=True, help="mention_based_probabilities.json")
    ap.add_argument("--kg_in", required=True, help="输入 KG JSONL")
    ap.add_argument("--kg_out", required=True, help="输出 KG JSONL")
    ap.add_argument("--round", type=int, default=6, help="概率保留小数位")
    ap.add_argument("--create_nodes", action="store_true", help="未命中时是否创建节点（默认不创建）")
    ap.add_argument("--cond_rel", default="presents_as", help="疾病->症状 条件概率关系名")
    ap.add_argument("--joint_rel", default="co_occurs_with", help="症状<->症状 联合概率关系名")
    return ap.parse_args()

# ---------------------- I/O ----------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------------------- utils ----------------------
def norm_en_key(s: str) -> str:
    """忽略大小写与空格（可加 NFKC/轻微清洗，但按你的需求仅去空格&lower）"""
    if not s:
        return ""
    return re.sub(r"\s+", "", s.strip().lower())

def parse_ent_no(ent_id: str):
    m = re.match(r"ENT_(\d+)$", ent_id or "")
    return int(m.group(1)) if m else None

# ---------------------- KG ----------------------
def load_kg(kg_path):
    nodes, rels = [], []
    for obj in load_jsonl(kg_path):
        if obj.get("type") == "relation":
            rels.append(obj)
        else:
            nodes.append(obj)
    return nodes, rels

def index_nodes(nodes):
    """
    分别建立：
      disease_idx: {norm_en: id}  for type=="Disease"
      symptom_idx: {norm_en: id}  for type=="Symptom"
    并记录最大的 ENT 序号
    """
    disease_idx, symptom_idx = {}, {}
    max_no = 0
    for n in nodes:
        nid = n.get("id")
        tno = parse_ent_no(nid)
        if tno is not None and tno > max_no:
            max_no = tno
        en = (n.get("text") or "").strip()
        if not en:
            continue
        key = norm_en_key(en)
        if n.get("type") == "Disease":
            disease_idx[key] = nid
        elif n.get("type") == "Symptom":
            symptom_idx[key] = nid
    return disease_idx, symptom_idx, max_no

def ensure_node(en_text, typ, idx_map, nodes, max_no, allow_create=False):
    key = norm_en_key(en_text)
    nid = idx_map.get(key)
    if nid:
        return nid, max_no
    if not allow_create:
        return None, max_no
    max_no += 1
    nid = f"ENT_{max_no:06d}"
    nodes.append({"type": typ, "id": nid, "text": en_text, "attr": {}})
    idx_map[key] = nid
    return nid, max_no

def index_relations(rels):
    m = {}
    for r in rels:
        k = f"{r.get('head')}||{r.get('relation')}||{r.get('tail')}"
        m[k] = r
    return m

# ---------------------- main ----------------------
def main():
    args = get_args()

    # 1) 概率载入
    mbp = load_json(args.prob_path)
    D_prior = mbp.get("disease_prior_prob", {})  # {en_disease: p}
    S_prior = mbp.get("symptom_prior_prob", {})  # {en_symptom: p}
    P_S_given_D = mbp.get("cond_prob", {})       # {en_disease: {en_symptom: p}}
    P_SS = mbp.get("joint_prob", {})             # {en_symptom1: {en_symptom2: p}}

    # 2) KG & 索引
    nodes, rels = load_kg(args.kg_in)
    disease_idx, symptom_idx, max_no = index_nodes(nodes)
    rel_map = index_relations(rels)

    # 统计与未匹配
    matched = {
        "disease_priors": 0,
        "symptom_priors": 0,
        "cond_edges": 0,    # presents_as
        "joint_pairs": 0    # co_occurs_with 对数（成对写两条）
    }
    unmatched = {
        "disease_priors": [],   # {"en": ..., "reason": "node-not-found"}
        "symptom_priors": [],   # {"en": ..., "reason": "node-not-found"}
        "cond": [],             # {"d_en": ..., "s_en": ..., "reason": "..."}
        "joint": []             # {"s1_en": ..., "s2_en": ..., "reason": "..."}
    }

    # 3) 写 Disease 先验
    for en_d, p in D_prior.items():
        d_id, max_no = ensure_node(en_d, "Disease", disease_idx, nodes, max_no, allow_create=args.create_nodes)
        if not d_id:
            unmatched["disease_priors"].append({"en": en_d, "reason": "node-not-found"})
            continue
        for n in nodes:
            if n.get("id") == d_id:
                n.setdefault("attr", {})["prior"] = round(float(p), args.round)
                break
        matched["disease_priors"] += 1

    # 4) 写 Symptom 先验
    for en_s, p in S_prior.items():
        s_id, max_no = ensure_node(en_s, "Symptom", symptom_idx, nodes, max_no, allow_create=args.create_nodes)
        if not s_id:
            unmatched["symptom_priors"].append({"en": en_s, "reason": "node-not-found"})
            continue
        for n in nodes:
            if n.get("id") == s_id:
                n.setdefault("attr", {})["prior"] = round(float(p), args.round)
                break
        matched["symptom_priors"] += 1

    # 5) 写 P(S|D) -> presents_as
    for en_d, sym_dict in P_S_given_D.items():
        d_id, max_no = ensure_node(en_d, "Disease", disease_idx, nodes, max_no, allow_create=args.create_nodes)
        if not d_id:
            # 整个疾病都没命中
            for en_s, _p in sym_dict.items():
                unmatched["cond"].append({"d_en": en_d, "s_en": en_s, "reason": "disease-node-not-found"})
            continue

        for en_s, p in sym_dict.items():
            s_id, max_no = ensure_node(en_s, "Symptom", symptom_idx, nodes, max_no, allow_create=args.create_nodes)
            if not s_id:
                unmatched["cond"].append({"d_en": en_d, "s_en": en_s, "reason": "symptom-node-not-found"})
                continue
            k = f"{d_id}||{args.cond_rel}||{s_id}"
            obj = rel_map.get(k) or {"type":"relation","head": d_id,"relation": args.cond_rel,"tail": s_id,"attr": {}}
            obj.setdefault("attr", {})["weight"] = round(float(p), args.round)
            rel_map[k] = obj
            matched["cond_edges"] += 1

    # 6) 写 P(Si,Sj) -> co_occurs_with（对称边）
    #    这里不假设输入是对称的，统一按输入建双向，去重由 rel_map 主键控制
    for en_s1, d in P_SS.items():
        s1_id, max_no = ensure_node(en_s1, "Symptom", symptom_idx, nodes, max_no, allow_create=args.create_nodes)
        if not s1_id:
            for en_s2, _p in d.items():
                unmatched["joint"].append({"s1_en": en_s1, "s2_en": en_s2, "reason": "symptom1-node-not-found"})
            continue
        for en_s2, p in d.items():
            s2_id, max_no = ensure_node(en_s2, "Symptom", symptom_idx, nodes, max_no, allow_create=args.create_nodes)
            if not s2_id:
                unmatched["joint"].append({"s1_en": en_s1, "s2_en": en_s2, "reason": "symptom2-node-not-found"})
                continue
            w = round(float(p), args.round)
            for h, t in ((s1_id, s2_id), (s2_id, s1_id)):
                k = f"{h}||{args.joint_rel}||{t}"
                obj = rel_map.get(k) or {"type":"relation","head": h,"relation": args.joint_rel,"tail": t,"attr": {}}
                obj.setdefault("attr", {})["weight"] = w
                rel_map[k] = obj
            matched["joint_pairs"] += 1

    # 7) 输出
    rels_out = list(rel_map.values())
    write_jsonl(nodes + rels_out, args.kg_out)

    report_path = os.path.join(os.path.dirname(args.kg_out), "unmatched_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(unmatched, f, ensure_ascii=False, indent=2)

    # 8) 汇总
    print("=== DONE ===")
    print(f"KG 输出: {args.kg_out}")
    print(f"未匹配报告: {report_path}")
    print(f"节点总数: {len(nodes)}  关系总数: {len(rels_out)}")
    print("匹配统计：")
    print(f"  Disease priors 写入: {matched['disease_priors']}")
    print(f"  Symptom priors 写入: {matched['symptom_priors']}")
    print(f"  presents_as 边数   : {matched['cond_edges']}")
    print(f"  co_occurs_with 对数: {matched['joint_pairs']}")
    print("未匹配统计：")
    print(f"  disease_priors 未匹配: {len(unmatched['disease_priors'])}")
    print(f"  symptom_priors 未匹配: {len(unmatched['symptom_priors'])}")
    print(f"  cond 未匹配          : {len(unmatched['cond'])}")
    print(f"  joint 未匹配         : {len(unmatched['joint'])}")

if __name__ == "__main__":
    main()
