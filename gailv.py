#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, unicodedata, re, itertools, collections
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

# ---------------- 工具函数 ----------------

CH_PUNCT = "，。！？、；：（）【】《》“”‘’—…·"
RE_WS_PUNCT = re.compile(rf"[{re.escape(CH_PUNCT)}\s]+")

def norm(s: str) -> str:
    """温和规范化：NFKC + 去两端空白 + 去连续空白/中文标点；不改动中文大小写。"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = RE_WS_PUNCT.sub("", s)
    return s

def safe_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    return [x]

def read_excel_disease_set(xlsx_path: Path) -> Tuple[Dict[str,str], List[str]]:
    """
    返回:
      sel_map: 规范化名 -> 显示名（来自Excel“病例库匹配数据_*”列）
      cols_used: 实际抓取到的“病例库匹配数据_*”列名
    """
    df = pd.read_excel(xlsx_path)
    # 找所有“病例库匹配数据_*”列
    match_cols = [c for c in df.columns if str(c).startswith("病例库匹配数据")]
    if not match_cols:
        raise ValueError("Excel 中未找到任何 '病例库匹配数据_*' 列")

    sel_map: Dict[str,str] = {}
    for c in match_cols:
        for v in df[c].dropna().astype(str):
            vv = v.strip()
            if not vv: continue
            sel_map.setdefault(norm(vv), vv)  # 用Excel里出现的原样词作显示名
    return sel_map, match_cols

def iter_records_from_json_dir(json_dir: Path):
    """遍历目录下所有 .json，yield 每条 RECORD（附文件来源）。"""
    for fp in sorted(json_dir.glob("**/*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ 解析失败，跳过: {fp} ({e})")
            continue
        recs = data.get("RECORDS", [])
        if isinstance(recs, list):
            for r in recs:
                yield fp, r
        else:
            print(f"⚠️ 非期望结构（无 RECORDS 列表），跳过: {fp}")

# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="包含 '病例库匹配数据_*' 列的 Excel 文件路径")
    ap.add_argument("--json_dir", required=True, help="病例库 JSON 文件夹（递归读取 *.json）")
    ap.add_argument("--output_dir", default="./case_prob_out", help="输出目录")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 从 Excel 收集要保留的疾病集合
    sel_map, used_cols = read_excel_disease_set(Path(args.xlsx))  # norm_name -> display_name
    sel_norm_set = set(sel_map.keys())
    print(f"✅ Excel 读取完成：匹配列 {len(used_cols)} 个，候选疾病 {len(sel_norm_set)} 个")

    # 2) 过滤病例：只保留含目标疾病的病例，并裁剪 final_disease
    kept_records: List[Dict[str, Any]] = []
    for src_fp, rec in iter_records_from_json_dir(Path(args.json_dir)):
        final_dis = [d for d in safe_list(rec.get("final_disease")) if isinstance(d, str) and d.strip()]
        if not final_dis:
            continue
        # 裁剪到目标疾病集合
        kept_displays = []
        for d in final_dis:
            dn = norm(d)
            if dn in sel_norm_set:
                kept_displays.append(sel_map[dn])  # 统一显示名为Excel版本
        if not kept_displays:
            continue  # 该病例不含目标疾病，丢弃

        # 准备输出一条“裁剪后”的记录
        new_rec = dict(rec)  # 浅拷贝即可
        new_rec["final_disease"] = sorted(set(kept_displays))  # 去重 + 排序
        # 规范化症状：仅去两端空白（不做激进归一）
        final_sym = [s.strip() for s in safe_list(rec.get("final_symptom")) if isinstance(s, str) and s.strip()]
        new_rec["final_symptom"] = sorted(set(final_sym))
        new_rec["_source_file"] = str(src_fp)
        kept_records.append(new_rec)

    if not kept_records:
        print("❗未保留到任何病例。请检查 Excel 疾病名称与病例库中的名称是否一致。")
        return

    # 保存过滤后的病例（便于复核）
    filtered_path = out_dir / "filtered_cases.jsonl"
    with filtered_path.open("w", encoding="utf-8") as f:
        for r in kept_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ 已输出过滤后的病例：{filtered_path}（{len(kept_records)} 条）")

    # 3) 统计并计算概率
    N = len(kept_records)  # 概率分母：保留病例总数（多标签允许∑P(D)>1）

    # 3.1 疾病先验 P(D)
    disease_counts = collections.Counter()
    for r in kept_records:
        disease_counts.update(r["final_disease"])
    disease_prior = {d: disease_counts[d] / N for d in sorted(disease_counts)}

    # 3.2 症状先验 P(S)
    symptom_counts = collections.Counter()
    for r in kept_records:
        symptom_counts.update(set(r["final_symptom"]))  # 一条病例内重复只计一次
    symptom_prior = {s: symptom_counts[s] / N for s in sorted(symptom_counts)}

    # 3.3 条件概率 P(S|D)
    #   对每个疾病，计算其出现的病例数作分母；分子为同时含该疾病且症状出现的病例数
    cond_counts: Dict[str, collections.Counter] = {d: collections.Counter() for d in disease_counts}
    for r in kept_records:
        sym_set = set(r["final_symptom"])
        dis_set = set(r["final_disease"])
        for d in dis_set:
            cond_counts[d].update(sym_set)
    cond_prob_long = []  # 长表
    cond_prob_wide = {}  # 宽表：disease -> {symptom: prob}
    for d in sorted(disease_counts):
        denom = disease_counts[d]
        row = {}
        for s in sorted(symptom_counts):
            num = cond_counts[d][s]
            p = (num / denom) if denom > 0 else 0.0
            row[s] = p
            cond_prob_long.append({"disease": d, "symptom": s, "p_s_given_d": p, "num": num, "den": denom})
        cond_prob_wide[d] = row

    # 3.4 症状-症状联合概率 P(S_i, S_j)
    pair_counts = collections.Counter()
    for r in kept_records:
        sym_list = sorted(set(r["final_symptom"]))
        for a, b in itertools.combinations(sym_list, 2):
            pair_counts[(a, b)] += 1
    pair_prob_long = []
    for (a, b), c in sorted(pair_counts.items()):
        pair_prob_long.append({"symptom_i": a, "symptom_j": b, "p_joint": c / N, "count": c, "den": N})

    # 4) 导出结果
    # 4.1 疾病先验
    df_dp = pd.DataFrame([{"disease": d, "count": disease_counts[d], "p_d": disease_prior[d]} for d in sorted(disease_prior)])
    df_dp.to_csv(out_dir / "disease_prior.csv", index=False, encoding="utf-8-sig")

    # 4.2 症状先验
    df_sp = pd.DataFrame([{"symptom": s, "count": symptom_counts[s], "p_s": symptom_prior[s]} for s in sorted(symptom_prior)])
    df_sp.to_csv(out_dir / "symptom_prior.csv", index=False, encoding="utf-8-sig")

    # 4.3 条件概率（长/宽）
    df_cond_long = pd.DataFrame(cond_prob_long)
    df_cond_long.to_csv(out_dir / "cond_p_s_given_d_long.csv", index=False, encoding="utf-8-sig")

    # 宽表：行=疾病，列=症状
    df_cond_wide = pd.DataFrame.from_dict(cond_prob_wide, orient="index").reset_index().rename(columns={"index":"disease"})
    df_cond_wide.to_csv(out_dir / "cond_p_s_given_d_wide.csv", index=False, encoding="utf-8-sig")

    # 4.4 症状联合概率
    df_pair = pd.DataFrame(pair_prob_long)
    df_pair.to_csv(out_dir / "symptom_joint_prob.csv", index=False, encoding="utf-8-sig")

    # 4.5 也存一份 JSON 便于程序读
    (out_dir / "probabilities.json").write_text(json.dumps({
        "meta": {
            "num_records": N,
            "note": "多标签计数：同一病例含多个疾病则每个疾病都计一次。"
        },
        "disease_prior": df_dp.to_dict(orient="records"),
        "symptom_prior": df_sp.to_dict(orient="records"),
        "cond_p_s_given_d": df_cond_long.to_dict(orient="records"),
        "symptom_joint": df_pair.to_dict(orient="records"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ 概率已计算完成")
    print(f"📄 输出目录：{out_dir}")
    print("  - disease_prior.csv")
    print("  - symptom_prior.csv")
    print("  - cond_p_s_given_d_long.csv / cond_p_s_given_d_wide.csv")
    print("  - symptom_joint_prob.csv")
    print("  - filtered_cases.jsonl（过滤后的病例）")
    print("  - probabilities.json（汇总 JSON）")

if __name__ == "__main__":
    main()
