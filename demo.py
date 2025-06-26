import sys
import os
import re
# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lpo_optimizer.core.optimization import run_lpo
from lpo_optimizer.core.evaluation import evaluate_prompt
from lpo_optimizer.utils.data_loader import load_bbh_dataset
import time

def extract_ans_tag(text):
    match = re.search(r"<ans>(.*?)</ans>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().lower()
    return text.strip().lower()

def test_prompt_on_samples(prompt: str, dataset, num_samples=5):
    """在少量样本上测试提示效果"""
    from lpo_optimizer.config import TASK_MODEL, OPENAI_API_KEY, OPENAI_API_BASE_URL
    from openai import OpenAI

    client = OpenAI(
        base_url=OPENAI_API_BASE_URL,
        api_key=OPENAI_API_KEY
    )
    
    print(f"\n=== 测试提示: '{prompt}' ===")
    correct = 0
    for i, (x, y_true) in enumerate(dataset[:num_samples]):
        try:
            response = client.chat.completions.create(
                model=TASK_MODEL,
                messages=[{"role": "user", "content": f"{prompt}\n\n{x}"}]
            )
            y_pred = response.choices[0].message.content.strip()
            y_pred_ans = extract_ans_tag(y_pred)
            is_correct = y_pred_ans == y_true.strip().lower()
            if is_correct:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"样本 {i+1}: {status}")
            print(f"  输入: {x}")
            print(f"  期望: {y_true}")
            print(f"  预测: {y_pred}")
            print(f"  提取答案: {y_pred_ans}")
            print()
            
        except Exception as e:
            print(f"样本 {i+1}: ❌ 错误 - {e}")
            print()
    
    accuracy = correct / num_samples
    print(f"准确率: {accuracy:.2%} ({correct}/{num_samples})")
    return accuracy

def main():
    print("🚀 LPO 提示优化演示")
    print("=" * 50)
    
    # 加载数据集
    print("📊 加载数据集...")
    dataset = load_bbh_dataset("data/BIG-Bench-Hard/bbh/boolean_expressions.json")
    print(f"数据集大小: {len(dataset)} 条")
    
    # 初始提示
    initial_prompt = "Please answer the following question. Put ONLY the final answer in <ans>...</ans> tags."
    
    print(f"\n🎯 初始提示: '{initial_prompt}'")
    
    # 测试初始提示效果
    initial_accuracy = test_prompt_on_samples(initial_prompt, dataset, num_samples=5)
    
    # 评估初始提示在完整数据集上的表现
    print(f"\n📈 评估初始提示在完整数据集上的表现...")
    start_time = time.time()
    initial_full_accuracy = evaluate_prompt(initial_prompt, dataset[:20])  # 使用前20条数据
    initial_time = time.time() - start_time
    print(f"完整数据集准确率: {initial_full_accuracy:.2%}")
    print(f"评估耗时: {initial_time:.2f}秒")
    
    # 运行优化
    print(f"\n🔧 开始优化提示...")
    print("优化过程可能需要几分钟，请耐心等待...")
    start_time = time.time()
    optimized_prompt = run_lpo(initial_prompt, dataset[:20])  # 使用前20条数据优化
    optimization_time = time.time() - start_time
    
    print(f"\n✨ 优化完成!")
    print(f"优化耗时: {optimization_time:.2f}秒")
    print(f"优化后的提示: '{optimized_prompt}'")
    
    # 测试优化后提示效果
    optimized_accuracy = test_prompt_on_samples(optimized_prompt, dataset, num_samples=5)
    
    # 评估优化后提示在完整数据集上的表现
    print(f"\n📈 评估优化后提示在完整数据集上的表现...")
    start_time = time.time()
    optimized_full_accuracy = evaluate_prompt(optimized_prompt, dataset[:20])
    optimized_time = time.time() - start_time
    print(f"完整数据集准确率: {optimized_full_accuracy:.2%}")
    print(f"评估耗时: {optimized_time:.2f}秒")
    
    # 效果对比总结
    print(f"\n📊 优化效果总结")
    print("=" * 50)
    print(f"初始提示: '{initial_prompt}'")
    print(f"优化提示: '{optimized_prompt}'")
    print()
    print(f"样本测试准确率:")
    print(f"  初始: {initial_accuracy:.2%}")
    print(f"  优化: {optimized_accuracy:.2%}")
    print(f"  提升: {optimized_accuracy - initial_accuracy:+.2%}")
    print()
    print(f"完整数据集准确率:")
    print(f"  初始: {initial_full_accuracy:.2%}")
    print(f"  优化: {optimized_full_accuracy:.2%}")
    print(f"  提升: {optimized_full_accuracy - initial_full_accuracy:+.2%}")
    print()
    print(f"优化耗时: {optimization_time:.2f}秒")
    
    if optimized_full_accuracy > initial_full_accuracy:
        print("🎉 优化成功！准确率有所提升。")
    elif optimized_full_accuracy == initial_full_accuracy:
        print("➡️ 优化后准确率保持不变。")
    else:
        print("⚠️ 优化后准确率略有下降，可能需要调整参数。")

if __name__ == "__main__":
    main()
