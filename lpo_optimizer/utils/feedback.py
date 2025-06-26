from typing import List

def generate_feedback(incorrect_examples: List[str]) -> str:
    """
    根据错误示例生成结构化反馈
    Args:
        incorrect_examples: 错误示例列表
    Returns:
        自然语言反馈
    """
    error_types = {
        "math": "Ensure arithmetic steps are verified.",
        "logic": "Clarify transitional logic between steps."
    }
    
    # 简化的错误分类（实际可用LLM分析）
    feedback = []
    for error in incorrect_examples[:3]:
        if any(word in error for word in ["+", "-", "×", "÷"]):
            feedback.append(error_types["math"])
        else:
            feedback.append(error_types["logic"])
    
    return "Key issues to fix:\n- " + "\n- ".join(set(feedback))