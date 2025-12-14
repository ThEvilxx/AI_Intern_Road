from openai import OpenAI

# 1. 建立连接：指向你本地刚刚装好的 Ollama
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama', # 本地运行不需要真密钥，随便填
)

print("系统：正在连接本地 RTX 3050...")

# 2. 发送请求：这是所有大模型开发最核心的函数
response = client.chat.completions.create(
    model="qwen2.5:1.5b", # 必须和你刚才 ollama run 的名字一样
    messages=[
        {"role": "system", "content": "你是一个暴躁的程序员。"},
        {"role": "user", "content": "Hello World 是什么意思？"}
    ]
)

# 3. 打印结果
print("--------------------------------")
print(response.choices[0].message.content)
print("--------------------------------")