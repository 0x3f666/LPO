from openai import OpenAI
from typing import List
from ..config import OPTIMIZER_MODEL, TEMPERATURE, OPENAI_API_KEY, OPENAI_API_BASE_URL

client = OpenAI(
    base_url=OPENAI_API_BASE_URL,
    api_key=OPENAI_API_KEY
)

def add_edit_tags(prompt: str, incorrect_examples: List[str]) -> str:
    """
    根据错误示例标记提示中需优化的词元（添加<edit>标签）
    Args:
        prompt: 初始提示（如"Let's think step by step"）
        incorrect_examples: 模型预测错误的示例列表
    Returns:
        带<edit>标签的提示
    """
    instruction = f"""
    Analyze these incorrect examples and identify tokens in the prompt to optimize.
    Wrap ONLY these tokens with <edit> and </edit> tags. Keep other parts unchanged.
    
    Incorrect Examples:
    {chr(10).join(incorrect_examples[:3])}
    
    Prompt:
    {prompt}
    """
    
    response = client.chat.completions.create(
        model=OPTIMIZER_MODEL,
        messages=[{"role": "user", "content": instruction}],
        temperature=TEMPERATURE
    )
    return response.choices[0].message.content