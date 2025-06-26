from openai import OpenAI
from ..config import OPTIMIZER_MODEL, TEMPERATURE, OPENAI_API_KEY, OPENAI_API_BASE_URL

# print(OPENAI_API_BASE_URL)

client = OpenAI(
    base_url=OPENAI_API_BASE_URL,
    api_key=OPENAI_API_KEY
)

def generate_local_prompt(tagged_prompt: str, feedback: str) -> str:
    """
    基于带<edit>标签的提示生成优化后的新提示
    Args:
        tagged_prompt: 带<edit>标签的提示
        feedback: 自然语言反馈
    Returns:
        优化后的提示（移除标签）
    """
    instruction = f"""
    Optimize ONLY text between <edit> tags based on feedback.
    Return the new prompt WITHOUT <edit> tags.
    
    Feedback:
    {feedback}
    
    Tagged Prompt:
    {tagged_prompt}
    """
    
    response = client.chat.completions.create(
        model=OPTIMIZER_MODEL,
        messages=[{"role": "user", "content": instruction}],
        temperature=TEMPERATURE
    )
    return response.choices[0].message.content