import json
import time
import os
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from zai import ZhipuAiClient

# === 配置 ===
API_KEY = "106350c189a5410cd95897207e4eafa2.Giu1p55HaVP6RCGu"
INPUT_BASE = "/home/bmm-system/data/private/yangjianxin/data/demo_medical_records"
OUTPUT_SUFFIX = "translated_records"
REJECTED_SUFFIX = "rejected_records"
ERROR_SUFFIX = "error_records"
FIELDS_TO_TRANSLATE = ["main_symptom", "final_disease", "final_symptom"]
MAX_WORKERS = 20
START_IDX, END_IDX = 1, 12

client = ZhipuAiClient(api_key=API_KEY)

# === 中文检测函数 ===
def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

# === 翻译函数（强prompt限制）===
def translate_text(text):
    try:
        response = client.chat.completions.create(
            model="glm-4.5",
            messages=[{
                "role": "user",
                "content": f"请将以下中文内容翻译成英文，要求专业、准确、流畅。请仅返回英文翻译结果，不要包含任何解释、注释、提示、重复内容或其他非翻译信息：\n\n{text}"
            }],
            thinking={"type": "enabled"},
            stream=False,
            temperature=0.3,
            max_tokens=1566,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] 翻译失败: {error_msg}")
        raise RuntimeError(error_msg)

# === 单条记录处理 ===
def process_record(index_record_tuple):
    idx, record = index_record_tuple
    contains_cn = False
    has_error = False

    try:
        for field in FIELDS_TO_TRANSLATE:
            if field in record:
                val = record[field]
                if isinstance(val, str):
                    translated = translate_text(val)
                    if contains_chinese(translated):
                        contains_cn = True
                    print(f"[Record {idx}] {field}: {val} → {translated}")
                    record[field] = translated
                elif isinstance(val, list):
                    translated_list = []
                    for i, item in enumerate(val):
                        translated_item = translate_text(item)
                        if contains_chinese(translated_item):
                            contains_cn = True
                        print(f"[Record {idx}] {field}[{i}]: {item} → {translated_item}")
                        translated_list.append(translated_item)
                    record[field] = translated_list
    except Exception as e:
        has_error = True
    return idx, record, contains_cn, has_error

# === 主流程：处理多个文件 ===
for i in range(START_IDX, END_IDX + 1):
    input_file = os.path.join(INPUT_BASE, f"demo{i}.json")
    output_file = os.path.join(INPUT_BASE, f"{OUTPUT_SUFFIX}{i}.jsonl")
    rejected_file = os.path.join(INPUT_BASE, f"{REJECTED_SUFFIX}{i}.jsonl")
    error_file = os.path.join(INPUT_BASE, f"{ERROR_SUFFIX}{i}.jsonl")

    print(f"\n====== 正在处理文件: {input_file} ======")

    # === 读取输入数据 ===
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        records = data["RECORDS"] if isinstance(data, dict) and "RECORDS" in data else data

    # === 跳过已翻译记录 ===
    translated_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f_out:
            for line in f_out:
                try:
                    parsed = json.loads(line.strip())
                    translated_ids.add(parsed.get("id"))
                except:
                    continue

    tasks = []
    for idx, record in enumerate(records):
        record_id = record.get("id")
        if record_id not in translated_ids:
            tasks.append((idx, record.copy()))
        else:
            print(f"[SKIP] 已翻译记录 id={record_id}")

    # === 并发翻译处理 ===
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_record, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc=f"翻译进度 demo{i}", unit="条"):
            idx, record, has_chinese, has_error = future.result()
            if has_error:
                with open(error_file, "a", encoding="utf-8") as f_err:
                    f_err.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[Record {idx}] ❌ 翻译报错 → 已记录到 {error_file}")
            else:
                target_file = rejected_file if has_chinese else output_file
                with open(target_file, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                if has_chinese:
                    print(f"[Record {idx}] ⚠️ 含中文 → 已记录到 {rejected_file}")

    print(f"✅ demo{i}.json 翻译完成，共处理 {len(tasks)} 条新记录，结果保存到：")
    print(f"    ✔️ 成功翻译：{output_file}")
    print(f"    ⚠️ 含中文记录：{rejected_file}")
    print(f"    ❌ 错误记录：{error_file}")
