#!/usr/bin/env python
# -*- coding: utf-8 -*-

from re import T 

import re
import json
import time
import os
from pathlib import Path

import openai

#########################################################
# 1. 读取 API KEY 和 BASE_URL  
#########################################################

with open('/mnt/workspace/xintong/api_key.txt', 'r', encoding="utf-8") as f:
    lines = f.readlines()

API_KEY = lines[0].strip()
BASE_URL = lines[1].strip()

openai.api_key = API_KEY
openai.base_url = BASE_URL

#########################################################
# 2. 路径配置
#########################################################

# 数据根目录（你要把 yo_train_task2.json / yo_large_eval_task2.json 放在这里）
DATA_ROOT = "/mnt/workspace/xintong/dataset/wc_yoruba"

TRAIN_JSON = os.path.join(DATA_ROOT, "yo_train_task2.json")
EVAL_JSON = os.path.join(DATA_ROOT, "yo_large_eval_task2.json")

# 输出根目录：结果应该放在 All_result/wc_yoruba/
OUTPUT_ROOT = "/mnt/workspace/xintong/jlq/All_result/wc_yoruba"
Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

#########################################################
# 3. 翻译用的提示词（system prompt）
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

举例输入：
{
  "opt_1": "India",
  "opt_2": "Mexico",
  "open_answer": "India"
}

举例输出（格式示意）：
{
  "opt_1": "Ìndíà",
  "opt_2": "Mẹ́sííkò",
  "open_answer": "Ìndíà"
}
"""

#########################################################
# 4. 工具函数：调用 API
#########################################################

def call_translation_api(items_dict, model_name, retries=3, retry_wait=2):
    """
    items_dict: 一个 dict，例如
    {
        "opt_1": "Ghana",
        "opt_2": "Vietnam",
        ...
        "open_answer": "India"
    }
    返回：翻译后的 dict
    """
    user_content = json.dumps(items_dict, ensure_ascii=False)

    for attempt in range(1, retries + 1):
        try:
            response = openai.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            text = response.choices[0].message.content

            # 期望模型返回的就是 JSON
            translated = json.loads(text)
            return translated

        except Exception as e:
            print(f"[API] 第 {attempt} 次调用失败：{e}")
            if attempt < retries:
                time.sleep(retry_wait)
            else:
                print("[API] 已重试多次仍失败，返回原始 items_dict 以避免崩溃。")
                return items_dict


#########################################################
# 5. 从 multi_choice_prompt 中提取选项，并重建题目
#########################################################

OPTION_PATTERN = re.compile(r'^(\d+)\.\s*(.+)$')

def extract_options_from_prompt(multi_choice_prompt):
    """
    输入：multi_choice_prompt（字符串）
    输出：
        header_lines: 选项之前的行列表
        options: [(line_index, option_number, option_text), ...]
        footer_lines: 选项之后的行列表
    """
    lines = multi_choice_prompt.splitlines()
    option_indices = []

    for idx, line in enumerate(lines):
        if OPTION_PATTERN.match(line.strip()):
            option_indices.append(idx)

    if not option_indices:
        # 没有匹配到选项，直接返回整段作为 header，其他为空
        return lines, [], []

    first_idx = min(option_indices)
    last_idx = max(option_indices)

    header_lines = lines[:first_idx]
    footer_lines = lines[last_idx + 1:]

    options = []
    for idx in range(first_idx, last_idx + 1):
        line = lines[idx].strip()
        m = OPTION_PATTERN.match(line)
        if m:
            num = m.group(1)
            text = m.group(2)
            options.append((idx, num, text))

    return header_lines, options, footer_lines


def rebuild_multi_choice_prompt(header_lines, options, footer_lines, translated_texts):
    """
    用翻译后的选项文本重建 multi_choice_prompt。
    translated_texts: dict, key 形如 "opt_1", "opt_2" ...
    """
    new_option_lines = []
    for _, num, _ in options:
        key = f"opt_{num}"
        new_text = translated_texts.get(key, "")  # 找不到就给个空字符串
        new_option_lines.append(f"{num}. {new_text}")

    # 拼回去
    all_lines = header_lines + new_option_lines + footer_lines
    return "\n".join(all_lines)


#########################################################
# 6. 主处理函数：翻译一个 JSON 文件
#########################################################

def process_json_file(input_path, model_name):
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {input_path}, total {len(data)} items.")

    for qa_id, item in data.items():
        mc_prompt = item.get("multi_choice_prompt", "")
        answer = item.get("answer", "")

        # 从题目中抽取选项
        header_lines, options, footer_lines = extract_options_from_prompt(mc_prompt)
        if not options:
            # 没有选项的题目（很少见），直接跳过
            continue

        # 组织需要翻译的内容
        to_translate = {}
        for _, num, text in options:
            key = f"opt_{num}"
            to_translate[key] = text

        # 有些题目可能没有 answer 字段，这里加个保护
        if isinstance(answer, str) and answer.strip():
            to_translate["open_answer"] = answer

        # 调 API 翻译
        translated = call_translation_api(to_translate, model_name=model_name)

        # 重建 multi_choice_prompt
        new_mc_prompt = rebuild_multi_choice_prompt(header_lines, options, footer_lines, translated)
        item["multi_choice_prompt"] = new_mc_prompt

        # 更新 answer
        if "open_answer" in translated:
            item["answer"] = translated["open_answer"]

    # 保存结果（注意输出目录已经换成 jlq / wc_yoruba）
    output_path = Path(OUTPUT_ROOT) / f"{model_name}_yo_translated_{input_path.name}"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved translated file to: {output_path}")


#########################################################
# 7. 入口
#########################################################

if __name__ == "__main__":

    model_name = "gpt-5-2025-08-07-GlobalStandard"
    print("Using model:", model_name)
    print("输出目录:", OUTPUT_ROOT)
    print("Train JSON:", TRAIN_JSON)
    print("Eval  JSON:", EVAL_JSON)

    # 依次处理 train 和 eval
    process_json_file(TRAIN_JSON, model_name)
    process_json_file(EVAL_JSON, model_name)
