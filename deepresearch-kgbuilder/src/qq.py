import uuid
import os
import asyncio
import time
from open_deep_research import __version__ as odr_version
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.graph import builder

# 设置 API 密钥
os.environ["ANTHROPIC_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

# 保存路径（可自定义）
SAVE_DIR = "/Users/yangjianxin/Downloads/disease_reports"
os.makedirs(SAVE_DIR, exist_ok=True)

# Prompt 模板
BASE_PROMPT = """Please systematically collect and organize professional veterinary knowledge specific to companion animals (dogs and cats) about {DISEASE_NAME}. The content must be strictly limited to small animal veterinary practice, excluding any human medical information. Cover at minimum the following structured sections:

1. **Disease Overview**
   - Definition
   - Epidemiological background

2. **Common Pathogens**
   - Viral and bacterial causes
   - Others if relevant

3. **Clinical Symptoms and Signs**

4. **Diagnostic Methods**
   - Clinical presentation
   - Laboratory testing (PCR, cultures, etc.)

5. **Treatment Options**
   - Medications
   - Supportive care
   - Nursing approaches

6. **Preventive Measures**
   - Vaccination
   - Isolation / environmental management

7. **Differential Diagnosis**
   - From other diseases with similar symptoms

8. **References**
   - Each knowledge point must cite its source (website URL, publication name, retrieval date)
   - Prioritize authoritative sources: Merck Veterinary Manual, VetLexicon, PubMed, WSAVA, AAHA

### Output Format
- Format in structured **Markdown** (or Word)
- Use clear section headers and a **numbered reference list**
- If any section lacks data or shows conflicting evidence, raise sub-questions and attempt to complete the information

→ Begin the task."""

async def generate_report(disease_name: str):
    print(f"\n🚀 Generating report for: {disease_name}")
    
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    pet_disease_name = f"pet disease {disease_name}"
    topic = f"Pet disease: {disease_name}"

    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "search_api": "tavily",
            "planner_provider": "anthropic",
            "planner_model": "claude-3-haiku-20240307",
            "writer_provider": "anthropic",
            "writer_model": "claude-3-haiku-20240307",
            "max_search_depth": 1,
            "number_of_queries": 1,
            "report_structure": BASE_PROMPT.format(DISEASE_NAME=pet_disease_name),
        }
    }

    # Step 1: 初次生成
    async for event in graph.astream({"topic": topic}, thread, stream_mode="updates"):
        if '__interrupt__' in event:
            break

    # Step 2: 提交反馈
    async for event in graph.astream(Command(resume=True), thread, stream_mode="updates"):
        if '__interrupt__' in event:
            break

    # Step 3: 继续生成最终报告
    async for _ in graph.astream(Command(resume=True), thread, stream_mode="updates"):
        pass

    # Step 4: 获取最终报告
    final_state = graph.get_state(thread)
    report = final_state.values.get("final_report")

    if report:
        filename = os.path.join(SAVE_DIR, f"{disease_name.replace('/', '_')}.md")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ Report saved: {filename}")
    else:
        print(f"❌ Failed to generate report for: {disease_name}")

async def main():
    # 仅处理一个疾病
    disease = "Aelurostrongylus Abstrusus Infection"  # ← 你可以改成任意疾病名
    try:
        await generate_report(disease)
    except Exception as e:
        print(f"⚠️ Error processing {disease}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
