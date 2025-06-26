from typing import List, Tuple
import json

def load_math_dataset(path: str) -> List[Tuple[str, str]]:
    """
    加载数学推理数据集（GSM8K格式）
    Args:
        path: JSON文件路径
    Returns:
        [(question, answer)]
    """
    with open(path) as f:
        data = json.load(f)
    return [(item["question"], item["answer"]) for item in data]

def load_bbh_dataset(path: str) -> List[Tuple[str, str]]:
    """
    加载BBH格式数据集（如boolean_expressions.json）
    Args:
        path: JSON文件路径
    Returns:
        [(input, target)]
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return [(item["input"], item["target"]) for item in data["examples"]]