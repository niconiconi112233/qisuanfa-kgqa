#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将知识图谱里 类型为“症状”的节点名称，按 Excel 映射表进行标准化替换；
并同步修正关系中以“名称字符串”表示的端点（若存在）。

用法示例：
python sync_symptom_names.py \
  --kg_in kg.json \
  --xlsx mapping.xlsx \
  --kg_out kg.synced.json \
  --sheet_name Sheet1 \
  --make_alias \
  --dry-run

参数说明：
- --kg_in        输入图谱 JSON 路径（必须）
- --xlsx         Excel 文件路径（必须）
- --kg_out       输出图谱 JSON 路径（省略则在原路径后加 .synced.json）
- --sheet_name   Excel 工作表名（缺省为第一个 sheet）
- --dry-run      只打印报告，不写回文件
- --make_alias   将旧名追加进实体的“别名”字段（若没有则创建）
"""

import json
import argparse
import os
from copy import deepcopy
from typing import Dict, Tuple, List, Any

import pandas as pd


# ========== 配置：关系端点可能的键名（按优先顺序探测） ==========
SOURCE_NAME_KEYS = ["来源名称", "source_name", "来源", "起点", "head_name", "subject", "from", "源"]
TARGET_NAME_KEYS = ["目标名称", "target_name", "目标", "终点", "tail_name", "object", "to", "汇"]
REL_TYPE_KEYS   = ["关系类型", "predicate", "rel", "关系"]  # 仅用于报告更友好


def normalize_text(s: str) -> str:
    """标准化：去首尾空白、全角空格→半角、中文顿号等常见符号简化、统一大小写（此处不改大小写，中文为主）"""
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("\u3000", " ")  # 全角空格
    return s


def load_mapping(xlsx_path: str, sheet_name: str = None) -> Dict[str, str]:
    """从 Excel 读取映射：图谱症状 -> 最终输出（优先用“最终输出2”若非空）"""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    needed_cols_any = [["图谱症状", "最终输出"], ["图谱症状", "最终输出", "最终输出2"]]
    cols = set(df.columns.astype(str))

    ok = False
    for cand in needed_cols_any:
        if set(cand).issubset(cols):
            ok = True
            break
    if not ok:
        raise ValueError(f"Excel 缺少必要列。至少需要包含：'图谱症状' 和 '最终输出'（可选 '最终输出2'）。实际列：{list(cols)}")

    mapping = {}
    for _, row in df.iterrows():
        src = normalize_text(row.get("图谱症状", ""))
        if not src:
            continue
        dst2 = normalize_text(row.get("最终输出2", "")) if "最终输出2" in df.columns else ""
        dst1 = normalize_text(row.get("最终输出", ""))
        dst = dst2 if dst2 else dst1
        if not dst:
            # 没有有效目标就跳过
            continue
        mapping[src] = dst

    return mapping


def detect_relation_name_keys(relations: List[dict]) -> Tuple[str, str, str]:
    """自动探测关系里用于“名称字符串端点”的键名"""
    if not relations:
        return "", "", ""

    sample = relations[0]
    src_key = next((k for k in SOURCE_NAME_KEYS if k in sample), "")
    tgt_key = next((k for k in TARGET_NAME_KEYS if k in sample), "")
    rel_key = next((k for k in REL_TYPE_KEYS if k in sample), "")

    return src_key, tgt_key, rel_key


def sync_kg(kg: dict, mapping: Dict[str, str], make_alias: bool = False) -> Tuple[dict, List[str]]:
    """
    同步图谱：
    - 症状节点名称替换
    - 同步替换关系中以“名称字符串”表示的端点
    返回：(新图谱, 报告行列表)
    """
    report = []
    kg_new = deepcopy(kg)

    entities: List[dict] = kg_new.get("实体", [])
    relations: List[dict] = kg_new.get("关系", [])

    # 1) 收集 症状 实体并替换名称
    rename_map: Dict[str, str] = {}   # old_name -> new_name（仅症状）
    changed_count = 0

    for ent in entities:
        if ent.get("类型") != "症状":
            continue
        name_old = normalize_text(ent.get("名称", ""))
        if not name_old:
            continue

        # 直接 exact 匹配映射（已标准化）
        if name_old in mapping:
            name_new = mapping[name_old]
            if name_new and name_new != name_old:
                # 修改名称
                ent["名称"] = name_new
                changed_count += 1
                rename_map[name_old] = name_new

                # 处理别名
                if make_alias:
                    aliases = ent.get("别名")
                    if not aliases or not isinstance(aliases, list):
                        aliases = []
                    if name_old not in aliases:
                        aliases.append(name_old)
                    ent["别名"] = aliases

                report.append(f"[实体] 症状：'{name_old}' → '{name_new}'")

    if changed_count == 0:
        report.append("未发现需要更名的症状实体。")

    # 2) 同步修正关系里的端点名称（如果关系中用“名称字符串”）
    src_key, tgt_key, rel_key = detect_relation_name_keys(relations)
    if src_key and tgt_key:
        r_changed = 0
        for rel in relations:
            src_val = normalize_text(rel.get(src_key, ""))
            tgt_val = normalize_text(rel.get(tgt_key, ""))

            new_src = rename_map.get(src_val, src_val)
            new_tgt = rename_map.get(tgt_val, tgt_val)

            did = False
            if new_src != src_val:
                rel[src_key] = new_src
                did = True
            if new_tgt != tgt_val:
                rel[tgt_key] = new_tgt
                did = True

            if did:
                r_changed += 1
                rname = rel.get(rel_key, "") if rel_key else ""
                report.append(f"[关系] {src_val} -[{rname}]-> {tgt_val}  =>  {new_src} -[{rname}]-> {new_tgt}")

        if r_changed == 0 and changed_count > 0:
            report.append("关系中未发现名称端点需要同步更新（可能关系使用的是 ID 连接，或没有使用可识别的名称字段）。")
    else:
        if changed_count > 0:
            report.append("未检测到关系端点的‘名称’字段（如 '来源名称' / '目标名称'），跳过关系名称同步。若关系以 ID 指向则无需更名。")

    return kg_new, report


def main():
    ap = argparse.ArgumentParser(description="按 Excel 映射统一症状实体名称并同步关系端点。")
    ap.add_argument("--kg_in", required=True, help="输入知识图谱 JSON 路径（含 '实体'，可选 '关系'）")
    ap.add_argument("--xlsx", required=True, help="Excel 映射表路径，至少包含列：图谱症状、最终输出（可选：最终输出2）")
    ap.add_argument("--kg_out", default="", help="输出图谱 JSON 路径（默认在输入名后缀加 .synced.json）")
    ap.add_argument("--sheet_name", default=None, help="Excel 工作表名（默认第一个 sheet）")
    ap.add_argument("--dry-run", action="store_true", help="只打印变更报告，不写回文件")
    ap.add_argument("--make_alias", action="store_true", help="将旧名添加到实体 '别名' 列表中")
    args = ap.parse_args()

    kg_out = args.kg_out or (args.kg_in.rsplit(".", 1)[0] + ".synced.json")

    # 读取数据
    with open(args.kg_in, "r", encoding="utf-8") as f:
        kg = json.load(f)

    mapping = load_mapping(args.xlsx, args.sheet_name)

    kg_new, report = sync_kg(kg, mapping, make_alias=args.make_alias)

    # 打印报告
    print("\n=== 变更报告 ===")
    for line in report:
        print(line)

    # 统计
    ent_total = len(kg.get("实体", []))
    rel_total = len(kg.get("关系", [])) if isinstance(kg.get("关系"), list) else 0
    print(f"\n实体总数：{ent_total}；关系总数：{rel_total}")

    if args.dry_run:
        print("\n[Dry-Run] 预览模式，未写回文件。")
        return

    # 写回
    with open(kg_out, "w", encoding="utf-8") as f:
        json.dump(kg_new, f, ensure_ascii=False, indent=2)
    print(f"\n已写出：{kg_out}")


if __name__ == "__main__":
    main()
