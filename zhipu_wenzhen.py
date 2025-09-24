#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于“中文加权知识图谱(实体/关系)”的宠物问诊 —— 纯硬匹配版
- 不依赖任何 embedding / 向量库
- 症状匹配：只在图谱里按 规范名/别名 做硬匹配（全等/子串，可配置）
- 概率读取：实体.概率.p_d / p_s；关系.权重.p_s_given_d / p_joint
- 关系字段：起点 / 终点 / 关系类型
- 实体字段：类型 / 名称 / 概率 / (可选)别名

依赖：
  pip install zai
环境变量：
  ZHIPUAI_API_KEY=xxxx
"""

import os
import re
import json
import argparse
import logging
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any

from zai import ZhipuAiClient

# ---------------- Logging ----------------
logger = logging.getLogger("KGDiagCN-Hard")
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(console)

def norm_cn(s: str) -> str:
    """中文/通用名字规范化：去空白 + NFKC"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", "", s.strip())

# ---------------- LLM ----------------
class GLM45:
    def __init__(self, api_key: Optional[str] = None, model: str = "glm-4-plus", temperature: float = 0.3, max_tokens: int = 8192):
        api_key = api_key or os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("ZHIPUAI_APIKEY") or os.environ.get("ZHIPUAI_API_KEY")
        if not api_key:
            raise RuntimeError("未找到 ZHIPUAI_API_KEY，请设置环境变量或在GLM45()里传入。")
        self.client = ZhipuAiClient(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages, stream: bool=False, thinking_enabled: bool=True) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            thinking={"type":"enabled"} if thinking_enabled else None
        )
        if stream:
            out = []
            for chunk in resp:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    out.append(delta)
                    print(delta, end="", flush=True)
            print()
            return "".join(out)
        else:
            try:
                return resp.choices[0].message.content
            except Exception:
                out = []
                for chunk in resp:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        out.append(delta)
                return "".join(out)

    def extract_symptoms(self, complaint_cn: str) -> Dict[str, List[str]]:
        user_prompt = f"""
请从这段中文主诉中抽取症状短语，并分到三个列表：
- positive: 用户明确提到有
- negative: 用户明确提到没有/否认
- uncertain: 提到但不确定

严格输出JSON，键固定为：positive, negative, uncertain；列表元素为简短中文短语。
不得输出除JSON外任何字符。

主诉：
{complaint_cn}
"""
        text = self.chat([{"role":"user","content":user_prompt}], stream=False)
        text = text.strip()
        lb = text.find("{"); rb = text.rfind("}")
        if lb>=0 and rb>lb: text = text[lb:rb+1]
        try:
            data = json.loads(text)
        except Exception:
            data = {"positive":[], "negative":[], "uncertain":[]}
        for k in ("positive","negative","uncertain"):
            data.setdefault(k, [])
            data[k] = [s.strip() for s in data[k] if isinstance(s,str) and s.strip()]
        return data

    def ask_yesno(self, pet_name: str, pet_type: str, symptom_cn: str) -> str:
        q = f"请问{pet_type}{pet_name}是否有“{symptom_cn}”这种表现？（是/否/不确定）请简要描述。"
        print(f"\n[问诊] {q}")
        ans = input("> 用户：").strip()

        judge_prompt = f"""
问题：{q}
用户回答：{ans}

请判断该回答是否为“是”(y)、“否”(n) 或 “不确定”(u)。
严格只输出一个字符：y 或 n 或 u
"""
        res = self.chat([{"role":"user","content":judge_prompt}], stream=False)
        res = res.strip().lower()
        if "y" in res and not ("n" in res or "u" in res): return "y"
        if "n" in res and not ("y" in res or "u" in res): return "n"
        if "u" in res or ("不确定" in res): return "u"
        if any(k in ans for k in ["有","是","出现","明显","经常"]): return "y"
        if any(k in ans for k in ["无","没有","未见","否认"]): return "n"
        return "u"

# --- 口语化工具：把规范症状名 -> 家长易懂的中文短语（带缓存） ---
def verbalize_symptom_cn(glm: GLM45, term: str, cache: Dict[str, str]) -> str:
    """
    输入规范症状名（如“腹部膨大”）；
    输出口语化短语（如“肚子鼓起来”“肚子胀”），不加标点/引号。
    只在反问时使用；其他逻辑不变。
    """
    if term in cache:
        return cache[term]

    prompt = (
        "请把下面这个医学症状名改写成口语化、家长易懂的中文短语，"
        "长度约4-12个字；不要加任何解释、前后缀、标点或引号：\n" + term
    )
    out = glm.chat([{"role": "user", "content": prompt}], stream=False) or ""
    # 清理：去引号/标点，只取第一行，做个长度兜底
    spoken = re.sub(r"[\"“”‘’。！!?？,.、，]", "", out).strip().splitlines()[0] if out else term
    if not spoken:
        spoken = term
    if len(spoken) > 20:
        spoken = spoken[:20]
    cache[term] = spoken
    return spoken

# ---------------- 概率装载（中文图谱JSON） ----------------
class KGProbCN:
    """
    读取：
      - 疾病先验：实体[类型=疾病].概率.p_d
      - 症状先验：实体[类型=症状].概率.p_s
      - 条件概率：关系[关系类型==cond_rel_type].权重.p_s_given_d
      - 联合概率：关系[关系类型==joint_rel_type].权重.p_joint（内部按无序对存）
    """
    def __init__(self, kg_json_path: str, cond_rel_type: str = "疾病-症状", joint_rel_type: str = "症状共现"):
        self.kg_json_path = kg_json_path
        self.cond_rel_type = cond_rel_type
        self.joint_rel_type = joint_rel_type

        self.disease_prior: Dict[str, float] = {}
        self.symptom_prior: Dict[str, float] = {}
        self.psd: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.joint: Dict[Tuple[str, str], float] = {}

        # 供硬匹配使用的集合/别名映射
        self.symptom_names: Set[str] = set()
        self.alias_to_canonical: Dict[str, str] = {}

        self._load()

    def _add_alias(self, canonical: str, alias_val: Any):
        """把别名（字符串或列表）加入 alias->canonical 映射"""
        if alias_val is None:
            return
        if isinstance(alias_val, str):
            a = alias_val.strip()
            if a:
                self.alias_to_canonical[a] = canonical
        elif isinstance(alias_val, list):
            for x in alias_val:
                if isinstance(x, str):
                    a = x.strip()
                    if a:
                        self.alias_to_canonical[a] = canonical

    def _load(self):
        with open(self.kg_json_path, "r", encoding="utf-8") as f:
            kg = json.load(f)

        ents = kg.get("实体", []) or []
        rels = kg.get("关系", []) or []

        # 先验 + 症状名与别名表
        for e in ents:
            typ = (e.get("类型") or "").strip()
            name = (e.get("名称") or "").strip()
            prob = e.get("概率") or {}
            if not name:
                continue

            if typ == "疾病":
                if isinstance(prob, dict) and "p_d" in prob:
                    self.disease_prior[name] = float(prob["p_d"])
            elif typ == "症状":
                self.symptom_names.add(name)
                if isinstance(prob, dict) and "p_s" in prob:
                    self.symptom_prior[name] = float(prob["p_s"])
                # 别名
                self._add_alias(name, e.get("别名"))

        # 条件与联合
        for r in rels:
            rtype = (r.get("关系类型") or "").strip()
            h = (r.get("起点") or "").strip()
            t = (r.get("终点") or "").strip()
            w = r.get("权重") or {}
            if not isinstance(w, dict):
                continue
            if rtype == self.cond_rel_type:
                p = w.get("p_s_given_d", None)
                if p is None: 
                    continue
                # 只收录“图谱里已存在的疾病/症状”
                if h in self.disease_prior and t in self.symptom_prior:
                    self.psd[h][t] = float(p)
            elif rtype == self.joint_rel_type:
                p = w.get("p_joint", None)
                if p is None:
                    continue
                # 只收录“图谱里已存在的症状”
                if h in self.symptom_prior and t in self.symptom_prior:
                    a, b = sorted([h, t])
                    self.joint[(a, b)] = max(self.joint.get((a, b), 0.0), float(p))

# ---------------- 一跳视图（中文图谱JSON） ----------------
class KGViewCN:
    ALLOWED_TYPES = {"疾病","症状","治疗","病原体","宿主","诊断","预防","鉴别诊断"}

    def __init__(self, kg_json_path: str):
        with open(kg_json_path, "r", encoding="utf-8") as f:
            kg = json.load(f)
        self.entities: List[dict] = kg.get("实体", []) or []
        self.relations: List[dict] = kg.get("关系", []) or []

        self.nodes_by_name: Dict[str, dict] = {}
        self.out_edges: Dict[str, List[dict]] = defaultdict(list)
        self.in_edges: Dict[str, List[dict]]  = defaultdict(list)

        for e in self.entities:
            typ = e.get("类型")
            name = e.get("名称")
            if not name:
                continue
            if typ in self.ALLOWED_TYPES:
                self.nodes_by_name[name] = e

        for r in self.relations:
            h = r.get("起点"); t = r.get("终点")
            if h: self.out_edges[h].append(r)
            if t: self.in_edges[t].append(r)

    def get_one_hop_subgraph(self, disease_name_cn: str) -> Optional[dict]:
        if disease_name_cn not in self.nodes_by_name:
            return None
        disease_node = self.nodes_by_name[disease_name_cn]
        neighbors = []

        for r in self.out_edges.get(disease_name_cn, []):
            tail_name = r.get("终点")
            nb = self.nodes_by_name.get(tail_name, {})
            if not nb:
                continue
            neighbors.append({
                "direction": "out",
                "relation_type": r.get("关系类型"),
                "weight": r.get("权重") or {},
                "neighbor": nb
            })
        for r in self.in_edges.get(disease_name_cn, []):
            head_name = r.get("起点")
            nb = self.nodes_by_name.get(head_name, {})
            if not nb:
                continue
            neighbors.append({
                "direction": "in",
                "relation_type": r.get("关系类型"),
                "weight": r.get("权重") or {},
                "neighbor": nb
            })

        return {
            "disease_node": disease_node,
            "neighbors": neighbors
        }

# ---------------- 症状“硬匹配检索” ----------------
class SymptomMatcherCN:
    """
    仅在图谱里按 规范名/别名 做硬匹配：
      1) 全等（规范名优先，其次别名）
      2) （可选）子串包含（规范名优先，其次别名）
    返回：命中的“规范名”集合
    """
    def __init__(self, symptom_names: Set[str], alias_to_canonical: Dict[str, str], enable_substring: bool = True):
        self.symptom_names = set(symptom_names)
        self.alias_to_canonical = dict(alias_to_canonical)
        self.enable_substring = enable_substring

        # 规范化索引
        self.norm_canonical = {norm_cn(x): x for x in self.symptom_names}
        self.norm_alias = {norm_cn(a): c for a, c in self.alias_to_canonical.items()}

    def match_phrases(self, phrases: List[str]) -> Dict[str, List[str]]:
        """
        返回：{用户短语: [命中的规范名,...]}（去重，顺序：全等>子串）
        """
        result: Dict[str, List[str]] = {}
        for q in phrases:
            q0 = q.strip()
            if not q0:
                continue
            qn = norm_cn(q0)

            hits_exact: List[str] = []
            hits_sub: List[str] = []

            # 1) 全等：规范名
            if qn in self.norm_canonical:
                hits_exact.append(self.norm_canonical[qn])
            # 2) 全等：别名
            if qn in self.norm_alias:
                hits_exact.append(self.norm_alias[qn])

            # 3/4) 子串
            if self.enable_substring and not hits_exact:
                # 规范名子串
                for cn in self.symptom_names:
                    if qn and norm_cn(cn).find(qn) != -1:
                        hits_sub.append(cn)
                # 别名字串
                if not hits_sub:
                    for a, c in self.alias_to_canonical.items():
                        if qn and norm_cn(a).find(qn) != -1:
                            hits_sub.append(c)

            # 去重并写入（全等优先）
            uniq = []
            seen = set()
            for lst in (hits_exact, hits_sub):
                for name in lst:
                    if name not in seen:
                        uniq.append(name)
                        seen.add(name)

            if uniq:
                result[q0] = uniq
        return result

# ---------------- Inference（直接中文名） ----------------
class InferenceCN:
    def __init__(self, kgprob: KGProbCN, negative_alpha: float = 0.5):
        self.kg = kgprob
        self.negative_alpha = negative_alpha
        self.post = {d: self.kg.disease_prior.get(d, 1e-8) for d in self.kg.disease_prior.keys()}
        self.S_pos: List[str] = []
        self.S_neg: List[str] = []

    def _p_s(self, s: str) -> float:
        return self.kg.symptom_prior.get(s, 1e-8)

    def _p_s_given_d(self, s: str, d: str) -> float:
        return self.kg.psd.get(d, {}).get(s, 1e-8)

    def _p_s_given_sj(self, s: str, sj: str) -> float:
        if s == sj:
            return 1.0
        a, b = sorted([s, sj])
        p_joint = self.kg.joint.get((a, b), 0.0)
        p_sj = self._p_s(sj)
        if p_sj <= 0:
            return 1e-8
        return min(max(p_joint / p_sj, 1e-8), 1.0)

    def _denom_for_positive(self, s: str) -> float:
        if len(self.S_pos) == 0:
            return self._p_s(s)
        return max(self._p_s_given_sj(s, sj) for sj in self.S_pos) or 1e-8

    def _update_positive(self, s: str):
        denom = self._denom_for_positive(s)
        for d in list(self.post.keys()):
            num = self._p_s_given_d(s, d)
            self.post[d] = self.post[d] * (num / max(denom, 1e-8))
            self.post[d] = min(max(self.post[d], 0.0), 1.0)
        if s not in self.S_pos:
            self.S_pos.append(s)

    def _update_negative(self, s: str):
        for d in list(self.post.keys()):
            num = (1.0 - self._p_s_given_d(s, d))
            den = (1.0 - self._p_s(s))
            factor = min(max(num / max(den, 1e-8), 0.0), 1.0)
            factor = self.negative_alpha * factor + (1 - self.negative_alpha) * 1.0
            self.post[d] = self.post[d] * factor
            self.post[d] = min(max(self.post[d], 0.0), 1.0)
        if s not in self.S_neg:
            self.S_neg.append(s)

    def incorporate(self, pos_sym: List[str], neg_sym: List[str]):
        for s in pos_sym: self._update_positive(s)
        for s in neg_sym: self._update_negative(s)
        self._renorm()

    def _renorm(self):
        s = sum(self.post.values())
        if s > 0:
            for k in list(self.post.keys()):
                self.post[k] = self.post[k] / s

    def top_diseases(self, k=5) -> List[Tuple[str, float]]:
        return sorted(self.post.items(), key=lambda x: x[1], reverse=True)[:k]

    def build_candidate_from_topd(self, top_diseases: List[str], per_disease: int = 15) -> List[str]:
        cand = []
        for d in top_diseases:
            sv = self.kg.psd.get(d, {})
            top_s = sorted(sv.items(), key=lambda x: x[1], reverse=True)[:per_disease]
            cand.extend([s for s, _ in top_s])
        seen = set()
        out = []
        for s in cand:
            if s in self.S_pos or s in self.S_neg: 
                continue
            if s not in seen:
                out.append(s); seen.add(s)
        return out

    def propose_next_symptom(self, full_pool: List[str], topk_disease_for_pool: int = 5, per_disease: int = 15) -> Optional[str]:
        topd = [d for d, _ in self.top_diseases(k=topk_disease_for_pool)]
        cand = self.build_candidate_from_topd(topd, per_disease=per_disease)
        if not cand:
            scores = []
            for s in full_pool:
                if s in self.S_pos or s in self.S_neg:
                    continue
                score = 0.0
                for d, pd in self.post.items():
                    score += pd * self._p_s_given_d(s, d)
                scores.append((s, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            cand = [s for s, _ in scores[:50]]
        if not cand:
            return None

        best_s, best_gain = None, -1e9
        baseline = sorted(self.post.values(), reverse=True)[0]
        denom_cache = {}

        for s in cand:
            snapshot = self.post.copy()
            denom = denom_cache.get(s)
            if denom is None:
                denom = self._denom_for_positive(s)
                denom_cache[s] = denom
            for d in list(snapshot.keys()):
                num = self._p_s_given_d(s, d)
                snapshot[d] = snapshot[d] * (num / max(denom, 1e-8))
            ssum = sum(snapshot.values()) or 1.0
            for d in snapshot:
                snapshot[d] /= ssum
            gain = sorted(snapshot.values(), reverse=True)[0] - baseline
            if gain > best_gain:
                best_gain, best_s = gain, s

        if best_s is not None and best_gain > 0:
            return best_s

        scores = []
        for s in cand:
            sc = 0.0
            for d, pd in self.post.items():
                sc += pd * self._p_s_given_d(s, d)
            scores.append((s, sc))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores else None

# --------- LLM 生成（使用中文子图） ---------
def generate_visit_advice(glm: GLM45, disease_cn: str, subgraph: dict) -> str:
    dump_json = json.dumps(subgraph, ensure_ascii=False, indent=2)
    prompt = f"""
你是资深小动物临床医生。下面给出疾病的一跳知识子图(JSON)。请输出：
- visit_advice_cn：中文就诊建议长文（友好、条理化）。
- 需把邻居按类别与关系名**分组罗列**：症状(基于 关系类型='疾病-症状' 且按权重排序)、治疗、诊断检查、病原体、宿主/易感、预防、鉴别诊断等。
- 可引用 knowledge_dump 中的引用/来源字段（若有）。
- **只输出严格 JSON**：{{"visit_advice_cn":"..."}}

knowledge_dump（原样，不要更改）：
{dump_json}
"""
    out = glm.chat([{"role":"user","content":prompt}], stream=False)
    txt = (out or "").strip()
    lb = txt.find("{"); rb = txt.rfind("}")
    if lb>=0 and rb>lb:
        txt = txt[lb:rb+1]
    return txt

def generate_disease_summary(glm: GLM45, disease_cn: str, subgraph: dict) -> str:
    dump_json = json.dumps(subgraph, ensure_ascii=False, indent=2)
    prompt = f"""
请基于下方疾病的一跳知识子图，产出**中文疾病总结**，覆盖：概述（中文名：{disease_cn}）、典型症状(按 '疾病-症状' 权重排序)、病因/病原体、受累解剖/宿主因素、常用检查与诊断要点、治疗思路、预后、家庭护理与复诊建议、预防、鉴别诊断、红旗征象。若有 reference/来源 请以“可参考资料”提及。  
**只输出严格 JSON**：{{"disease_summary_cn":"..."}}

知识子图（原样，不要改动）：
{dump_json}
"""
    out = glm.chat([{"role":"user","content":prompt}], stream=False)
    txt = (out or "").strip()
    lb = txt.find("{"); rb = txt.rfind("}")
    if lb>=0 and rb>lb:
        txt = txt[lb:rb+1]
    return txt

# ---------------- Runner ----------------
def run_dialog(
    kg_path: str,
    pet_name: str = "毛毛",
    pet_type: str = "猫",
    positive_conf_threshold: float = 0.75,
    max_turns: int = 5,
    save_advice_dir: Optional[str] = None,
    cond_rel_type: str = "疾病-症状",
    joint_rel_type: str = "症状共现",
    enable_substring_match: bool = True,
):
    # 载入KG概率 & 视图（中文）
    kgp = KGProbCN(kg_path, cond_rel_type=cond_rel_type, joint_rel_type=joint_rel_type)
    kgv = KGViewCN(kg_path)
    if not kgp.disease_prior or not kgp.symptom_prior:
        logger.warning("KG中未发现 prior 概率，推断可能不稳定。")

    # 症状全集（来自图谱“症状实体”与/或 p_s 键）
    all_symptoms = sorted(set(kgp.symptom_prior) | set(kgp.symptom_names))

    # 硬匹配器
    matcher = SymptomMatcherCN(
        symptom_names=set(all_symptoms),
        alias_to_canonical=kgp.alias_to_canonical,
        enable_substring=enable_substring_match
    )

    glm = GLM45()
    spoken_cache: Dict[str, str] = {}  # 仅用于反问时的口语化短语缓存

    # 1) 主诉
    main_complaint = input("您好呀，请简单描述一下您家宠物的情况：\n> ").strip()
    # 2) LLM 抽取中文症状
    extracted = glm.extract_symptoms(main_complaint)
    pos_cn = extracted.get("positive", [])
    neg_cn = extracted.get("negative", [])

    # 3) 硬匹配：短语 -> 规范症状名
    pos_map = matcher.match_phrases(pos_cn) if pos_cn else {}
    neg_map = matcher.match_phrases(neg_cn) if neg_cn else {}

    # 统计展示
    print("\n[匹配结果-阳性]")
    if pos_map:
        for q, names in pos_map.items():
            print(f"  '{q}' -> {names}")
    else:
        print("  （无）")

    print("[匹配结果-阴性]")
    if neg_map:
        for q, names in neg_map.items():
            print(f"  '{q}' -> {names}")
    else:
        print("  （无）")

    def flatten_hits(hit_map: Dict[str, List[str]]) -> List[str]:
        out, seen = [], set()
        for _, lst in hit_map.items():
            for name in lst:
                if name not in seen:
                    out.append(name); seen.add(name)
        return out

    pos_syms = flatten_hits(pos_map)
    neg_syms = flatten_hits(neg_map)

    # 4) 初始化推断
    engine = InferenceCN(kgp, negative_alpha=0.5)
    engine.incorporate(pos_syms, neg_syms)

    candidate_pool = all_symptoms[:]
    turn = 0
    final_disease_cn = None

    while turn < max_turns:
        top5 = engine.top_diseases(k=5)
        print("\n当前Top疾病：", [f"{d}: {p:.4f}" for d,p in top5])

        top1_dis, top1_p = top5[0]
        if top1_p >= positive_conf_threshold:
            print(f"\n达到置信阈值：{top1_dis} (prob={top1_p:.3f})")
            final_disease_cn = top1_dis
            break

        nxt_sym = engine.propose_next_symptom(candidate_pool, topk_disease_for_pool=5, per_disease=15)
        if not nxt_sym:
            print("没有更合适的追问症状，结束。")
            final_disease_cn = top1_dis
            break

        # —— 仅在反问时进行口语化改写，其余逻辑不变 ——
        spoken_sym = verbalize_symptom_cn(glm, nxt_sym, spoken_cache)
        ans = glm.ask_yesno(pet_name, pet_type, symptom_cn=spoken_sym)

        if ans == "y":
            engine.incorporate([nxt_sym], [])
        elif ans == "n":
            engine.incorporate([], [nxt_sym])

        candidate_pool = [s for s in candidate_pool if s != nxt_sym]
        turn += 1

    print("\n=== 最终推荐疾病 ===")
    for d, p in engine.top_diseases(k=5):
        print(f"{d}: {p:.4f}")

    # 生成一跳“就诊建议 / 疾病总结”
    if final_disease_cn:
        subgraph = kgv.get_one_hop_subgraph(final_disease_cn)
        if not subgraph:
            print(f"[WARN] 在KG中找不到疾病节点：{final_disease_cn}，跳过就诊建议与疾病总结。")
            return

        advice_json = generate_visit_advice(glm, final_disease_cn, subgraph)
        print("\n====== 就诊建议（LLM生成） ======")
        print(advice_json)

        summary_json = generate_disease_summary(glm, final_disease_cn, subgraph)
        print("\n====== 疾病总结（LLM生成） ======")
        print(summary_json)

        if save_advice_dir:
            os.makedirs(save_advice_dir, exist_ok=True)
            safe_name = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9_.-]+", "_", final_disease_cn)

            advice_path = os.path.join(save_advice_dir, f"advice_{safe_name}.json")
            with open(advice_path, "w", encoding="utf-8") as f:
                f.write(advice_json)
            print(f"[OK] 已保存就诊建议：{advice_path}")

            summary_path = os.path.join(save_advice_dir, f"summary_{safe_name}.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_json)
            print(f"[OK] 已保存疾病总结：{summary_path}")

# ---------------- CLI ----------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg_path", required=True, help="你的中文图谱 JSON 文件路径（含实体/关系/概率/权重）")
    ap.add_argument("--pet_name", default="毛毛")
    ap.add_argument("--pet_type", default="猫")
    ap.add_argument("--conf_th", type=float, default=0.75)
    ap.add_argument("--max_turns", type=int, default=5)
    ap.add_argument("--save_advice_dir", default="", help="保存就诊建议/疾病总结JSON的目录（可选）")
    ap.add_argument("--cond_rel_type", default="疾病-症状", help="图谱中 疾病->症状 的关系类型名")
    ap.add_argument("--joint_rel_type", default="症状共现", help="图谱中 症状<->症状 的关系类型名")
    ap.add_argument("--no_substring", action="store_true", help="只做全等匹配（关闭子串命中）")
    return ap.parse_args()

if __name__ == "__main__":
    args = get_args()
    run_dialog(
        kg_path=args.kg_path,
        pet_name=args.pet_name,
        pet_type=args.pet_type,
        positive_conf_threshold=args.conf_th,
        max_turns=args.max_turns,
        save_advice_dir=args.save_advice_dir or None,
        cond_rel_type=args.cond_rel_type,
        joint_rel_type=args.joint_rel_type,
        enable_substring_match=(not args.no_substring),
    )
