from typing import List, Tuple
from .edit_tagging import add_edit_tags
from .prompt_proposal import generate_local_prompt
from .evaluation import evaluate_prompt
from ..utils.feedback import generate_feedback
from ..config import MAX_OPTIMIZATION_STEPS, TASK_MODEL, OPENAI_API_KEY, OPENAI_API_BASE_URL
from openai import OpenAI

client = OpenAI(
    base_url=OPENAI_API_BASE_URL,
    api_key=OPENAI_API_KEY
)

def run_lpo(initial_prompt: str, dataset: List[Tuple[str, str]]) -> str:
    """
    LPO主优化流程
    Args:
        initial_prompt: 初始提示
        dataset: 训练数据
    Returns:
        最优提示
    """
    current_prompt = initial_prompt
    best_prompt = current_prompt
    best_score = evaluate_prompt(current_prompt, dataset)
    
    for step in range(MAX_OPTIMIZATION_STEPS):
        # 1. 收集错误示例（避免重复API调用）
        incorrect_examples = []
        for x, y in dataset:
            response = client.chat.completions.create(
                model=TASK_MODEL,
                messages=[{"role": "user", "content": f"{current_prompt}\n\n{x}"}]
            )
            y_pred = response.choices[0].message.content.strip()
            print(y_pred)
            if y_pred != y.strip():
                incorrect_examples.append(f"Input: {x}\nOutput: {y_pred}\nExpected: {y}")
        
        if not incorrect_examples:
            break  # 无错误则终止
        
        # 2. 标记优化词元
        tagged_prompt = add_edit_tags(current_prompt, incorrect_examples)
        
        # 3. 生成反馈并优化
        feedback = generate_feedback(incorrect_examples)
        new_prompt = generate_local_prompt(tagged_prompt, feedback)
        
        # 4. 评估更新
        new_score = evaluate_prompt(new_prompt, dataset)
        if new_score > best_score:
            best_score, best_prompt = new_score, new_prompt
        current_prompt = new_prompt
    
    return best_prompt