#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于加权知识图谱的宠物问诊
(GLM-4.5 + 本地BGE + 中英桥接 + 一跳就诊建议/疾病总结 + 图传播融合)

新增：GraphReasoner（PPR）把“结构化图证据”融合到疾病后验中，避免只是一跳概率。
参数（本版默认）：
  - negative_alpha = 0.5          阴性融合的温和系数
  - gamma = 0.3                   PPR 重启率
  - beta_graph = 0.3              图证据与贝叶斯后验的线性融合权重
  - use_ss = True                 使用 Symptom-Symptom 共现边
  - s2d_via_lr = True             S→D 用似然比 (P(S|D)/P(S)) 放大
依赖：
  pip install -U FlagEmbedding faiss-cpu  或 faiss-gpu
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
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel
from zai import ZhipuAiClient

# ---------------- Logging ----------------
logger = logging.getLogger("KGDiag")
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(console)

# ---------------- Utils ----------------
def norm_en(s: str) -> str:
    """英文名用于匹配：去空白+小写+NFKC"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", "", s.strip().lower())

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def load_translation_pairs(path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    返回：
      en2zh_norm: { norm_en(english) -> chinese }
      zh2en:      { chinese -> english (原样) }
    """
    en2zh_norm, zh2en = {}, {}
    if not path or not os.path.exists(path):
        return en2zh_norm, zh2en
    for j in load_jsonl(path):
        en = (j.get("english") or "").strip()
        zh = (j.get("chinese") or "").strip()
        if not en or not zh:
            continue
        en2zh_norm[norm_en(en)] = zh
        zh2en[zh] = en
    return en2zh_norm, zh2en

# ---------------- LLM (GLM-4.5 via zai) ----------------
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

    # --------- 任务prompt ----------
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

    def translate_en2zh(self, en_text: str) -> str:
        prompt = f"请把以下医学症状或疾病名翻译成简洁、常见的中文表达，不要输出任何多余内容：\n{en_text}"
        out = self.chat([{"role":"user","content":prompt}], stream=False)
        return (out or "").strip()

# ---------------- KG & Probabilities ----------------
class KGProb:
    """
    从KG JSONL加载：
      - Disease/Symptom 节点：attr.prior -> P(D)/P(S)
      - relation presents_as: weight -> P(S|D)
      - relation co_occurs_with: weight -> P(Si,Sj)
    """
    def __init__(self, kg_path: str):
        self.kg_path = kg_path
        self.disease_prior: Dict[str, float] = {}
        self.symptom_prior: Dict[str, float] = {}
        self.psd: Dict[str, Dict[str,float]] = defaultdict(dict)
        self.joint: Dict[Tuple[str,str], float] = {}
        self._load()

    def _load(self):
        id2en = {}
        for obj in load_jsonl(self.kg_path):
            typ = obj.get("type")
            if typ in ("Disease","Symptom"):
                en = (obj.get("text") or "").strip()
                if not en: continue
                pid = obj.get("id")
                id2en[pid] = en
                prior = (obj.get("attr") or {}).get("prior")
                if prior is not None:
                    if typ == "Disease":
                        self.disease_prior[en] = float(prior)
                    elif typ == "Symptom":
                        self.symptom_prior[en] = float(prior)

        for obj in load_jsonl(self.kg_path):
            if obj.get("type") != "relation":
                continue
            h = obj.get("head"); t = obj.get("tail")
            rel = obj.get("relation")
            w = (obj.get("attr") or {}).get("weight")
            if w is None: 
                continue
            w = float(w)
            if rel == "presents_as":
                d_en = id2en.get(h,""); s_en = id2en.get(t,"")
                if d_en and s_en:
                    self.psd[d_en][s_en] = w
            elif rel == "co_occurs_with":
                s1 = id2en.get(h,""); s2 = id2en.get(t,"")
                if s1 and s2:
                    k = tuple(sorted([s1, s2]))
                    self.joint[k] = max(self.joint.get(k, 0.0), w)

# ------- KG 一跳视图（用于就诊建议/总结） -------
class KGView:
    def __init__(self, kg_path: str):
        self.kg_path = kg_path
        self.nodes_by_id: Dict[str, dict] = {}
        self.out_edges: Dict[str, List[dict]] = defaultdict(list)
        self.in_edges: Dict[str, List[dict]]  = defaultdict(list)
        self.dis_en2id: Dict[str, str] = {}
        self._load()

    def _load(self):
        for obj in load_jsonl(self.kg_path):
            if obj.get("type") in ("Disease","Symptom","Category","Pathogen","Anatomy","Sign","Treatment","Diagnosis","Prevention","DiseaseAlt","Host"):
                nid = obj.get("id")
                self.nodes_by_id[nid] = obj
                if obj.get("type") == "Disease":
                    en = (obj.get("text") or "").strip()
                    if en:
                        self.dis_en2id[norm_en(en)] = nid
        for obj in load_jsonl(self.kg_path):
            if obj.get("type") == "relation":
                h = obj.get("head"); t = obj.get("tail")
                self.out_edges[h].append(obj)
                self.in_edges[t].append(obj)

    def get_one_hop_subgraph(self, disease_en: str) -> Optional[dict]:
        did = self.dis_en2id.get(norm_en(disease_en))
        if not did:
            return None
        disease_node = self.nodes_by_id.get(did, {})
        neighbors = []

        for r in self.out_edges.get(did, []):
            tail = r.get("tail")
            nb = self.nodes_by_id.get(tail, {})
            neighbors.append({
                "direction": "out",
                "relation": r.get("relation"),
                "relation_attr": r.get("attr") or {},
                "neighbor": {
                    "id": nb.get("id"),
                    "type": nb.get("type"),
                    "text": nb.get("text"),
                    "attr": nb.get("attr") or {},
                    "reference": nb.get("reference", None)
                }
            })
        for r in self.in_edges.get(did, []):
            head = r.get("head")
            nb = self.nodes_by_id.get(head, {})
            neighbors.append({
                "direction": "in",
                "relation": r.get("relation"),
                "relation_attr": r.get("attr") or {},
                "neighbor": {
                    "id": nb.get("id"),
                    "type": nb.get("type"),
                    "text": nb.get("text"),
                    "attr": nb.get("attr") or {},
                    "reference": nb.get("reference", None)
                }
            })

        return {
            "disease_node": {
                "id": disease_node.get("id"),
                "type": disease_node.get("type"),
                "text": disease_node.get("text"),
                "attr": disease_node.get("attr") or {},
                "reference": disease_node.get("reference", None)
            },
            "neighbors": neighbors
        }

# ---------------- Disease name set for filtering ----------------
def load_disease_name_sets(kg_path: str, translated_diseases: Optional[str]) -> Tuple[Set[str], Set[str]]:
    disease_en_raw_set: Set[str] = set()
    for obj in load_jsonl(kg_path):
        if obj.get("type") == "Disease":
            en = (obj.get("text") or "").strip()
            if en:
                disease_en_raw_set.add(en)
    disease_en_norm_set: Set[str] = {norm_en(x) for x in disease_en_raw_set}
    if translated_diseases and os.path.exists(translated_diseases):
        en2zh_norm, _ = load_translation_pairs(translated_diseases)
        disease_en_norm_set |= set(en2zh_norm.keys())
    return disease_en_raw_set, disease_en_norm_set

# ---------------- Symptom Retriever (BGE + FAISS, 双语入库) ----------------
class SymptomRetriever:
    def __init__(self, model_path: str, en_symptoms: List[str], en2zh_norm: Dict[str,str], topk: int = 2):
        self.model = BGEM3FlagModel(model_path, use_fp16=True)
        self.topk = topk
        self.corpus_texts: List[str] = []
        self.corpus_canonical_en: List[str] = []
        for en in en_symptoms:
            self.corpus_texts.append(en)
            self.corpus_canonical_en.append(en)
            zh = en2zh_norm.get(norm_en(en))
            if zh:
                self.corpus_texts.append(zh)
                self.corpus_canonical_en.append(en)
        mat = self._encode(self.corpus_texts)
        self.index = faiss.IndexFlatIP(mat.shape[1])
        self.index.add(mat)

    def _encode(self, texts: List[str]) -> np.ndarray:
        out = self.model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        v = out["dense_vecs"]
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        return v.astype("float32")

    def search(self, phrases_cn: List[str], min_cos: float = 0.55) -> Dict[str, List[Tuple[str, float]]]:
        if not phrases_cn:
            return {}
        qv = self._encode(phrases_cn)
        raw_topk = self.topk * 5
        D, I = self.index.search(qv, raw_topk)
        res = {}
        for qi, phrase in enumerate(phrases_cn):
            agg = {}
            for idx, cos in zip(I[qi], D[qi]):
                if idx < 0 or cos < min_cos:
                    continue
                en = self.corpus_canonical_en[idx]
                agg[en] = max(agg.get(en, 0.0), float(cos))
            hits = sorted(agg.items(), key=lambda x: x[1], reverse=True)[: self.topk]
            res[phrase] = hits
        return res

# ---------------- Graph Reasoner (PPR over KG) ----------------
class GraphReasoner:
    """
    在 Disease–Symptom Symptom–Symptom二部图做重启随机游走，
    以 S+ 为重启源，输出结构得分；并可用于融合入后验。
    
    """
    def __init__(self, kg: KGProb, use_ss: bool = True, s2d_via_lr: bool = True, gamma: float = 0.3):
        self.gamma = gamma
        self.d_list = list(kg.disease_prior.keys())
        self.s_list = list(kg.symptom_prior.keys())
        self.d_idx = {d:i for i,d in enumerate(self.d_list)}
        self.s_idx = {s:i for i,s in enumerate(self.s_list)}

        nd, ns = len(self.d_list), len(self.s_list)
        A_ds = np.zeros((nd, ns), dtype=np.float32)
        for d, mp in kg.psd.items():
            if d not in self.d_idx: continue
            i = self.d_idx[d]
            for s, w in mp.items():
                j = self.s_idx.get(s, None)
                if j is not None:
                    A_ds[i,j] = max(float(w), 0.0)

        eps = 1e-8
        if s2d_via_lr:
            P_s = np.array([kg.symptom_prior.get(s, eps) for s in self.s_list], dtype=np.float32)
            A_sd = (A_ds / np.maximum(P_s[None, :], eps)).T  # (ns, nd)
        else:
            A_sd = A_ds.T.copy()

        if use_ss:
            A_ss = np.zeros((ns, ns), dtype=np.float32)
            for (s1, s2), w in kg.joint.items():
                i = self.s_idx.get(s1, None); j = self.s_idx.get(s2, None)
                if i is not None and j is not None:
                    A_ss[i,j] = A_ss[j,i] = max(float(w), 0.0)
            if A_ss.max() > 0:
                A_ss = 0.2 * (A_ss / A_ss.max())  # 缩放，防止淹没 D<->S
        else:
            A_ss = np.zeros((ns, ns), dtype=np.float32)

        top = np.concatenate([np.zeros((nd, nd), np.float32), A_ds], axis=1)
        bottom = np.concatenate([A_sd, A_ss], axis=1)
        A = np.concatenate([top, bottom], axis=0)  # (nd+ns, nd+ns)

        row_sum = A.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        self.T = A / row_sum
        self.nd, self.ns = nd, ns

    def ppr_from_symptoms(self, pos_s: List[str], neg_s: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        n = self.nd + self.ns
        r = np.zeros((n,), dtype=np.float32)
        idxs = [self.nd + self.s_idx[s] for s in pos_s if s in self.s_idx]
        if idxs:
            r[idxs] = 1.0 / len(idxs)
        else:
            r[:] = 1.0 / n  # 无阳性时避免 NaN

        # 处理阴性：把阴性症状的重启质量置零
        for s in neg_s:
            j = self.s_idx.get(s, None)
            if j is not None:
                r[self.nd + j] = 0.0
        if r.sum() == 0:
            r[:] = 1.0 / n

        pi = r.copy()
        for _ in range(40):
            pi = (1 - self.gamma) * (self.T.T @ pi) + self.gamma * r

        pi_d = pi[:self.nd]
        pi_s = pi[self.nd:]
        if pi_d.sum() > 0: pi_d = pi_d / pi_d.sum()
        if pi_s.sum() > 0: pi_s = pi_s / pi_s.sum()
        return pi_d, pi_s

# ---------------- Inference Engine (报告公式 + 图融合) ----------------
class Inference:
    """
    维护疾病后验 P(D|evidence)
    阳性 s：post[D] <- post[D] * P(s|D)/P*(s)，其中 P*(s) 用 co-occurrence 近似抑制冗余
    阴性 s：温和融合 (1-P(s|D))/(1-P(s))
    末尾：与图传播得分做线性融合（beta_graph）
    """
    def __init__(self, kgprob: KGProb, negative_alpha: float = 0.5,
                 graph_reasoner: Optional[GraphReasoner] = None, beta_graph: float = 0.3):
        self.kg = kgprob
        self.negative_alpha = negative_alpha
        self.graph_reasoner = graph_reasoner
        self.beta_graph = beta_graph
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
        k = tuple(sorted([s, sj]))
        p_joint = self.kg.joint.get(k, 0.0)
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

    def incorporate(self, pos_sym_en: List[str], neg_sym_en: List[str]):
        for s in pos_sym_en: self._update_positive(s)
        for s in neg_sym_en: self._update_negative(s)
        self._renorm()

        # —— 图传播融合 —— #
        if self.graph_reasoner and self.beta_graph > 0:
            pi_d, _ = self.graph_reasoner.ppr_from_symptoms(self.S_pos, self.S_neg)
            # 将 pi_d 映射到疾病名
            gscore = {}
            for d, i in self.graph_reasoner.d_idx.items():
                gscore[d] = float(pi_d[i])
            # 线性融合并归一
            for d in self.post:
                self.post[d] = (1 - self.beta_graph) * self.post[d] + self.beta_graph * gscore.get(d, 0.0)
            self._renorm()

    def _renorm(self):
        s = sum(self.post.values())
        if s > 0:
            for k in list(self.post.keys()):
                self.post[k] = self.post[k] / s

    def top_diseases(self, k=5) -> List[Tuple[str, float]]:
        return sorted(self.post.items(), key=lambda x: x[1], reverse=True)[:k]

    # ---- 基于Top疾病的候选症状集 ----
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
                out.append(s)
                seen.add(s)
        return out

    # ---- 选“期望提升”最大的症状；多级回退 ----
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

# --------- 生成“就诊建议”：用 1-hop 子图（不丢字段） ---------
def generate_visit_advice(glm: GLM45, disease_en: str, disease_zh: Optional[str], subgraph: dict) -> str:
    disease_cn = disease_zh or disease_en
    dump_json = json.dumps(subgraph, ensure_ascii=False, indent=2)
    prompt = f"""
你是资深小动物临床医生。下面给出疾病的“一跳知识子图”，其中包含**全部**实体/属性/关系，严禁遗漏任何字段。
请输出一个 JSON：
{{
  "visit_advice_cn": "<中文建议长文>"
}}
写作要点：
- 疾病概述（中英文名：{disease_cn} / {disease_en}）
- 典型症状（参考 relation='presents_as' 权重）
- 鉴别诊断/并发症/病原体/解剖部位等（按关系名分组）
- 建议检查与评估要点（若缺省则给通用建议）
- 初步处置/护理与就医时机（标注红旗征象）
- 若 knowledge_dump 中有 reference，请列为“可参考资料”
一跳子图：
{dump_json}
"""
    out = glm.chat([{"role":"user","content":prompt}], stream=False)
    txt = (out or "").strip()
    lb = txt.find("{"); rb = txt.rfind("}")
    if lb>=0 and rb>lb:
        txt = txt[lb:rb+1]
    return txt

# --------- 生成“疾病总结”：把 1-hop 子图喂给 LLM 输出中文总结 ---------
def generate_disease_summary(glm: GLM45, disease_en: str, disease_zh: Optional[str], subgraph: dict) -> str:
    disease_cn = disease_zh or disease_en
    dump_json = json.dumps(subgraph, ensure_ascii=False, indent=2)
    prompt = f"""
你是一名资深小动物临床医生。请基于给定疾病的一跳知识子图，输出**中文疾病总结**。
输出严格 JSON：
{{
  "disease_summary_cn": "……长文……"
}}
必须覆盖：病名（中/英）、典型症状（参考 presents_as 权重）、病因/病原体、受累解剖、危险因素、
并发症/鉴别诊断、检查与诊断要点、治疗思路与预后、家庭护理与复诊建议、红旗征象、可参考资料（如有）。
病名：{disease_cn} / {disease_en}
一跳子图：
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
    embed_model_path: str,
    translated_symptoms: Optional[str],
    translated_diseases: Optional[str],
    pet_name: str = "毛毛",
    pet_type: str = "猫",
    topk_retrieval: int = 2,
    min_cos: float = 0.55,
    positive_conf_threshold: float = 0.85,
    max_turns: int = 5,
    save_advice_dir: Optional[str] = None,
):
    # 载入KG概率 & 视图
    kgp = KGProb(kg_path)
    kgv = KGView(kg_path)
    if not kgp.disease_prior or not kgp.symptom_prior:
        logger.warning("KG中未发现 prior 概率，推断可能不稳定。")

    # 载入中英对照
    sym_en2zh_norm, _ = load_translation_pairs(translated_symptoms)
    dis_en2zh_norm, _ = load_translation_pairs(translated_diseases)

    # 疾病集合（用于过滤把疾病当症状）
    disease_en_raw_set, disease_en_norm_set = load_disease_name_sets(kg_path, translated_diseases)

    # 症状全集（来自 P(S)），过滤掉疑似疾病名
    all_symptoms_raw = list(kgp.symptom_prior.keys())
    filtered_symptom_list = [s for s in all_symptoms_raw if norm_en(s) not in disease_en_norm_set]
    

    # 检索器
    retriever = SymptomRetriever(
        embed_model_path,
        en_symptoms=filtered_symptom_list,
        en2zh_norm=sym_en2zh_norm,
        topk=topk_retrieval
    )

    # GLM
    glm = GLM45()

    # 1) 主诉
    main_complaint = input("您好呀，请简单描述一下您家宠物的情况：\n> ").strip()
    # 2) LLM 抽取症状(中文)
    extracted = glm.extract_symptoms(main_complaint)
    pos_cn = extracted.get("positive", [])
    neg_cn = extracted.get("negative", [])

    # 3) CN -> EN 检索映射
    pos_map = retriever.search(pos_cn, min_cos=min_cos) if pos_cn else {}
    neg_map = retriever.search(neg_cn, min_cos=min_cos) if neg_cn else {}

    def filter_hits_to_symptoms(hit_map: Dict[str, List[Tuple[str, float]]]) -> List[str]:
        ens = []
        for _, hits in hit_map.items():
            for en, _ in hits:
                if norm_en(en) in disease_en_norm_set:
                    continue
                ens.append(en)
        return sorted(set(ens))

    pos_en = filter_hits_to_symptoms(pos_map)
    neg_en = filter_hits_to_symptoms(neg_map)

    # 4) 初始化图推理器 + 推断引擎
    gr = GraphReasoner(kgp, use_ss=True, s2d_via_lr=True, gamma=0.3)
    engine = Inference(kgp, negative_alpha=0.5, graph_reasoner=gr, beta_graph=0.3)
    engine.incorporate(pos_en, neg_en)

    # 候选症状池
    candidate_pool = filtered_symptom_list[:]

    def en2zh_display(en_name: str) -> str:
        zh = sym_en2zh_norm.get(norm_en(en_name))
        if zh: return zh
        return glm.translate_en2zh(en_name) or en_name

    turn = 0
    final_disease_en = None

    while turn < max_turns:
        top5 = engine.top_diseases(k=5)
        disp = []
        for d, p in top5:
            zh = dis_en2zh_norm.get(norm_en(d))
            disp.append(f"{(zh or d)} / {d}: {p:.4f}")
        print("\n当前Top疾病：", disp)

        top1_dis, top1_p = top5[0]
        if top1_p >= positive_conf_threshold:
            final_zh = dis_en2zh_norm.get(norm_en(top1_dis))
            print(f"\n达到置信阈值：{(final_zh or top1_dis)} / {top1_dis} (prob={top1_p:.3f})")
            final_disease_en = top1_dis
            break

        nxt_en = engine.propose_next_symptom(candidate_pool, topk_disease_for_pool=5, per_disease=15)
        # 兜底：避免把疾病名当成症状问
        if nxt_en and norm_en(nxt_en) in disease_en_norm_set:
            nxt_en = None

        if not nxt_en:
            print("没有更合适的追问症状，结束。")
            final_disease_en = top1_dis
            break

        nxt_zh = en2zh_display(nxt_en)
        ans = glm.ask_yesno(pet_name, pet_type, symptom_cn=nxt_zh)
        if ans == "y":
            engine.incorporate([nxt_en], [])
        elif ans == "n":
            engine.incorporate([], [nxt_en])
        candidate_pool = [s for s in candidate_pool if s != nxt_en]
        turn += 1

    print("\n=== 最终推荐疾病（中/英） ===")
    for d, p in engine.top_diseases(k=5):
        zh = dis_en2zh_norm.get(norm_en(d))
        print(f"{(zh or d)} / {d}: {p:.4f}")

    # --------- 生成一跳“就诊建议” + “疾病总结” ----------
    if final_disease_en:
        disease_zh = dis_en2zh_norm.get(norm_en(final_disease_en))
        kgv = KGView(kg_path)
        subgraph = kgv.get_one_hop_subgraph(final_disease_en)
        if not subgraph:
            print(f"[WARN] 在KG中找不到疾病节点：{final_disease_en}，跳过就诊建议与疾病总结。")
            return

        advice_json = generate_visit_advice(glm, final_disease_en, disease_zh, subgraph)
        print("\n====== 就诊建议（LLM生成） ======")
        print(advice_json)

        summary_json = generate_disease_summary(glm, final_disease_en, disease_zh, subgraph)
        print("\n====== 疾病总结（LLM生成） ======")
        print(summary_json)

        if save_advice_dir:
            os.makedirs(save_advice_dir, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", final_disease_en)

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
    ap.add_argument("--kg_path", default="/home/bmm-system/data/private/yangjianxin/data/demo_medical_records/outputs/prob_kg.jsonl", help="包含概率的知识图谱 JSONL")
    ap.add_argument("--embed_model_path", default="/home/bmm-system/data/private/yangjianxin/data/embedding", help="本地BGE模型路径（如 'BAAI/bge-m3' 或目录）")
    ap.add_argument("--translated_symptoms", default="/home/bmm-system/data/private/yangjianxin/data/demo_medical_records/outputs/translated_symptoms.jsonl", help="translated_symptoms.jsonl")
    ap.add_argument("--translated_diseases", default="/home/bmm-system/data/private/yangjianxin/data/demo_medical_records/outputs/translated_diseases.jsonl", help="translated_diseases.jsonl")
    ap.add_argument("--pet_name", default="毛毛")
    ap.add_argument("--pet_type", default="猫")
    ap.add_argument("--topk_retrieval", type=int, default=2)
    ap.add_argument("--min_cos", type=float, default=0.55)
    ap.add_argument("--conf_th", type=float, default=0.75)
    ap.add_argument("--max_turns", type=int, default=5)
    ap.add_argument("--save_advice_dir", default="/home/bmm-system/data/private/yangjianxin/data/baogao", help="将就诊建议/疾病总结JSON另存到该目录（可选）")
    return ap.parse_args()

if __name__ == "__main__":
    args = get_args()
    run_dialog(
        kg_path=args.kg_path,
        embed_model_path=args.embed_model_path,
        translated_symptoms=args.translated_symptoms,
        translated_diseases=args.translated_diseases,
        pet_name=args.pet_name,
        pet_type=args.pet_type,
        topk_retrieval=args.topk_retrieval,
        min_cos=args.min_cos,
        positive_conf_threshold=args.conf_th,
        max_turns=args.max_turns,
        save_advice_dir=args.save_advice_dir,
    )
