from openai import OpenAI
from typing import List, Tuple
from ..config import TASK_MODEL, OPENAI_API_KEY, OPENAI_API_BASE_URL
import re

client = OpenAI(
    base_url=OPENAI_API_BASE_URL,
    api_key=OPENAI_API_KEY
)

def extract_ans_tag(text):
    match = re.search(r"<ans>(.*?)</ans>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().lower()
    return text.strip().lower()

def evaluate_prompt(prompt: str, dataset: List[Tuple[str, str]]) -> float:
    """
    评估提示在数据集上的准确率
    Args:
        prompt: 待评估提示
        dataset: [(input, expected_output)]
    Returns:
        准确率（0-1）
    """
    correct = 0
    for x, y_true in dataset:
        response = client.chat.completions.create(
            model=TASK_MODEL,
            messages=[{"role": "user", "content": f"{prompt}\n\n{x}"}]
        )
        y_pred = response.choices[0].message.content.strip()
        y_pred_ans = extract_ans_tag(y_pred)
        if y_pred_ans == y_true.strip().lower():
            correct += 1
    return correct / len(dataset)