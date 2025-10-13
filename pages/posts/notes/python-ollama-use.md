---
title: 如何对 Ollama 进行单轮对话与多轮对话
date: 2025-08-13
type: notes
---

# 单轮对话实现

```python
# 使用openai的代码风格调用ollama

from openai import OpenAI

try:
    # 创建客户端
    client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")

    # 发送请求
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "你好，请介绍下自己"}], model="llama3"
    )

    # 输出结果
    print(chat_completion.choices[0].message.content)

except Exception as e:
    print(f"错误: {e}")
```

# 多轮对话实现

```python
# 多轮对话
from openai import OpenAI


# 定义多轮对话方法
def run_chat_session():
    try:
        # 初始化客户端
        client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
        print("聊天开始！输入 'exit' 或 'quit' 退出")
        print("-" * 40)

        # 初始化对话历史
        chat_history = []

        # 启动对话循环
        while True:
            try:
                # 获取用户输入
                user_input = input("用户：").strip()
                if user_input.lower() in ["exit", "quit", "退出"]:
                    print("退出对话。")
                    break

                if not user_input:  # 处理空输入
                    print("请输入有效内容")
                    continue

                # 更新对话历史(添加用户输入)
                chat_history.append({"role": "user", "content": user_input})

                # 调用模型回答
                chat_completion = client.chat.completions.create(
                    messages=chat_history, model="llama3"
                )

                # 获取最新回答
                model_response = chat_completion.choices[0]
                print("AI:", model_response.message.content)
                print("-" * 40)

                # 更新对话历史（添加AI模型的回复）
                chat_history.append(
                    {"role": "assistant", "content": model_response.message.content}
                )

            except KeyboardInterrupt:
                print("\n用户中断对话")
                break
            except Exception as e:
                print(f"对话错误：{e}")
                print("继续对话，输入 'exit' 退出")
                continue

    except Exception as e:
        print(f"初始化错误: {e}")

if __name__ == "__main__":
    run_chat_session()
```
