# Qisuanfa-KGQA 知识图谱问答系统

本项目为 **绮算法** 系列中的知识图谱问答系统子模块集合，旨在构建基于结构化医学知识的智能问答能力。



---

## 📁 `deepresearch-kgbuilder`

**功能**：  
基于 deep research 报告自动生成**疾病知识图谱**，包括：

- 原始疾病研究报告生成
- 三元组抽取与知识构建
- 输出结构化 JSON/CSV 结果

**内容包括：**

- 三元组抽取脚本
- `src/`：deepresearch疾病报告生成源码及多轮数据结果（包含文件夹chubu、deep_research_result、deep_research0722、new_result、report0711等）
- 最终结构化知识结果

---

## 📁 `deduplication`

**功能**：  
对抽取得到的三元组数据进行**实体与语义级别的去重处理**，提升知识图谱质量。

**内容包括：**

- 去重算法实现（如文本相似度、嵌入匹配等）
- `results/`：去重结果

## 📁 `算法设计报告`

**功能**：  
整体智能问诊算法设计报告

**内容包括：**

- 结合知识图谱问答
- 基于贝叶斯的疾病推理




2025-9-24
open_deep_research-mian.zip文件夹，包含
1.完整的deep_research生成代码，运行路径为src/ceshi.py。完整代码已配置好，更新路径可直接运行。
2.最新版的deepresearch结果，路径/src/reports-out。
3.最新版的知识图谱，路径：/src/triple


问诊算法实现：
zhipu_wenzhen.py


---

