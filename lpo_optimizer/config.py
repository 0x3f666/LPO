# OpenAI API配置
OPENAI_API_KEY = "sk-TB5xahA0SAB8WZAD0KxZYQAf7IsFSb3gLK2aDp1ZNi7efr2p"  # 替换为实际密钥
OPENAI_API_BASE_URL = "https://api.openai-proxy.org/v1"  # 可自定义为你的baseurl
OPTIMIZER_MODEL = "deepseek-chat"        # 用于优化的模型
TASK_MODEL = "gpt-4o-mini"     # 用于任务执行的模型

# 实验参数
MAX_OPTIMIZATION_STEPS = 3       # 最大优化轮次
BEAM_SIZE = 4                    
TEMPERATURE = 0.3                # 生成温度（控制随机性）