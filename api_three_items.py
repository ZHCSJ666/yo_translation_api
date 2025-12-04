#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path
from tqdm import tqdm
import openai   # ✅ 使用 SDK，而不是 requests

#########################################################
# 1. 读取 API KEY 和 BASE_URL  
#########################################################

with open('/mnt/workspace/xintong/api_key.txt', 'r', encoding="utf-8") as f:
    lines = f.readlines()

API_KEY = lines[0].strip()
BASE_URL = lines[1].strip()    # ⚠️ 建议这里是类似 https://xiaoai.plus/v1 这样的 base，不要再带 /chat/completions

# 设置给 SDK
openai.api_key = API_KEY
openai.base_url = BASE_URL


#########################################################
# 2. 输入 / 输出路径配置
#########################################################

BASE_DIR = Path(__file__).resolve().parent

INPUT_JSON = BASE_DIR / "three_items.json"

OUTPUT_ROOT = Path("/mnt/workspace/xintong/jlq/All_result/wc_yoruba")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


#########################################################
# 3. system prompt
#########################################################

TRANSLATION_SYSTEM_PROMPT = """
你是一个专业的约鲁巴语（Yorùbá, 语言代号 “yo”）翻译助手。

现在你会收到一个 JSON 对象，其中的 value 是一些单词或短语，
包括国家名、地区名或简短答案（例如 "India", "South Africa" 等）。

你的任务：
1. 把每一个 value 翻译成标准的约鲁巴语写法（含声调和变音符号，如果常用的话）。
2. 如果该词本身已经是约鲁巴语写法，可以原样保留或做轻微规范化。
3. 保持 JSON 的键（key）不变，只修改 value。
4. 严格只输出 JSON，不要输出其它文字。
"""


#########################################################
# 4. 调用 API（SDK 版本：openai.chat.completions.create）
#########################################################

def call_translation_api_once(items_dict, model_name):
    """
    只调用一次 API，失败由外层重试逻辑处理。
    items_dict 形如：
    {
        "opt_1": "Ghana",
        "opt_2": "Vietnam",
        "open_answer": "India"
    }
    """
    user_content = json.dumps(items_dict, ensure_ascii=False)

    # ✅ 使用 SDK，而不是 requests.post
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    text = response.choices[0].message.content
    translated = json.loads(text)  # 如果不是合法 JSON，会在这里抛错
    return translated


#########################################################
# 5. 主处理函数（带进度条 & 重试）
#########################################################

def process_translations(input_path, model_name, retries=3, retry_wait=2):

    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"[INFO] Loaded {input_path}, total {total} items.")

    all_translations = {}

    for qa_id, to_translate in tqdm(data.items(), total=total, desc="Translating"):

        if not isinstance(to_translate, dict):
            continue

        translated = None

        for attempt in range(1, retries + 1):
            try:
                translated = call_translation_api_once(to_translate, model_name)
                break
            except Exception as e:
                print(f"[{qa_id}] 第 {attempt} 次调用失败：{e}")
                if attempt < retries:
                    time.sleep(retry_wait)
                else:
                    print(f"[{qa_id}] 已重试 {retries} 次仍失败 → 写回原始内容")
                    translated = to_translate

        all_translations[str(qa_id)] = translated

    # 保存结果
    output_path = OUTPUT_ROOT / f"{model_name}_yo_raw_translations_{input_path.name}"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_translations, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved translation results → {output_path}")


#########################################################
# 6. 入口
#########################################################

if __name__ == "__main__":

    model_name = "gpt-5-2025-08-07-GlobalStandard"

    print("Using model:", model_name)
    print("Input JSON:", INPUT_JSON)
    print("输出目录:", OUTPUT_ROOT)

    process_translations(INPUT_JSON, model_name)
