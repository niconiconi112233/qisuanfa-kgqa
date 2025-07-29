import os
import json
from tqdm import tqdm
from anthropic import Anthropic
client = Anthropic(api_key="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")  # 替换成你的 API Key，建议用 os.getenv("ANTHROPIC_API_KEY")



# 粗略估算 token 数（假设英文为主，中文稍少）
def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)

# Prompt 模板
base_prompt = """You are a veterinary medical information extraction expert. Please extract structured information from the following veterinary medical text in JSON Lines format, one JSON object per line.

【Task Instructions】
1. Extract all medical entities (in English or Chinese), including:
   - Disease
   - Symptom
   - Pathogen
   - Host
   - Anatomy
   - DiseaseAlt (Differential Diagnosis)
   - Diagnosis (Diagnostic Method)
   - Treatment (Treatment Plan)
   - Prevention (Preventive Measure)

2. Extract attributes for each entity (when applicable):
   - Disease: type, reversibility, pathogenesis, complications
   - Symptom: location, frequency, severity
   - Pathogen: source, mechanism of action
   - Host: breed, age group
   - Anatomy: anatomical location, physiological function
   - Diagnosis: principle, applicable stage, limitations
   - Treatment: method, indication, risks
   - Prevention: implementation, target group, expected effect

3. Extract semantic relations (triplets) between entities. Supported relation types:
   - causes (Pathogen → Disease)
   - presents_as (Disease → Symptom)
   - differential_diagnosis (Disease → DiseaseAlt)
   - infects (Disease or Pathogen → Host)
   - affects (Disease → Anatomy)
   - diagnosed_by (Disease → Diagnosis)
   - treated_by (Disease → Treatment)
   - prevented_by (Disease → Prevention)

4. For each **entity**, add an additional field "reference" to indicate the original source(s) of this knowledge.
   - If the input text contains a numbered source list (e.g., [1], [2]), please resolve those numbers to their full citation text using the Sources section at the end of the document.
   - Format "reference" as: "Title - URL", e.g., "WHO Campylobacter Fact Sheet - https://www.who.int/news-room/fact-sheets/detail/campylobacter"
   - If multiple sources apply, separate with semicolons.
   - If the reference is unclear or the source cannot be matched, use "Unknown"

【Output Format】
Each line must be a single valid JSON object.

- Entity example:
{"type": "entity", "entity_text": "Campylobacter", "entity_type": "Pathogen", "attr": {"source": "foodborne", "mechanism of action": "GI mucosa invasion"}, "reference": "WHO Campylobacter Fact Sheet - https://www.who.int/news-room/fact-sheets/detail/campylobacter"}

- Relation example:
{"type": "relation", "head": "Campylobacteriosis", "relation_type": "causes", "tail": "Diarrhea"}

【Requirements】
- Use English punctuation and double quotes.
- No markdown, no explanations, no arrays or nested JSON.
- Output only one JSON object per line.
- If no attributes, use "attr": {}, and always include the full "reference" string in each entity.
- Ensure sources like [1], [2] are resolved using the "Sources" section in the text.

【Input Text】
{text}

【Start Output】Please output one JSON object per line:
"""

# 文件路径
input_folder = "/Users/yangjianxin/Downloads/open_deep_research-main/src"
output_folder = "/Users/yangjianxin/Downloads/open_deep_research-main/src/tripe_outputs_0707"
os.makedirs(output_folder, exist_ok=True)

# 初始化统计变量
total_input_tokens = 0
total_output_tokens = 0

# 收集 .md 文件
md_files = [f for f in os.listdir(input_folder) if f.endswith(".md")]

# 主循环，带 tqdm
for filename in tqdm(md_files, desc="Processing files"):
    filepath = os.path.join(input_folder, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    prompt = base_prompt.replace("{text}", content)
    input_token_est = estimate_tokens(prompt)
    total_input_tokens += input_token_est

    # Claude 请求
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=8192,
        temperature=0.2,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    output_text = response.content[0].text.strip()
    output_token_est = estimate_tokens(output_text)
    total_output_tokens += output_token_est

    out_path = os.path.join(output_folder, filename.replace(".md", ".jsonl"))
    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write(output_text)

    print(f"✅ Processed: {filename} → {out_path}")

# 成本估算
input_cost = total_input_tokens / 1_000_000 * 15
output_cost = total_output_tokens / 1_000_000 * 75
total_cost = input_cost + output_cost

print(f"\n📊 Token & Cost Summary:")
print(f"🟡 Estimated Input Tokens: {int(total_input_tokens)} → ${input_cost:.2f}")
print(f"🟢 Estimated Output Tokens: {int(total_output_tokens)} → ${output_cost:.2f}")
print(f"💰 Total Estimated Cost: ${total_cost:.2f}")
