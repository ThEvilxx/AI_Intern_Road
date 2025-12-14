from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# 1. 初始化模型 (还是用你昨天下的那个)
print("正在唤醒 Qwen2.5...")
llm = OllamaLLM(model="qwen2.5:1.5b")

# 2. 加载本地数据
print("正在读取机密文件...")
# encoding='utf-8' 是为了防止 Windows 中文乱码
loader = TextLoader("./data/secret.txt", encoding='utf-8')
docs = loader.load()
secret_content = docs[0].page_content

# 3. 构建 RAG 原理 (手动拼接)
# 核心思想：把查到的资料强行塞到 Prompt 里告诉 AI
template = """
你是一个助手。请根据下面的【已知信息】来回答用户的问题。
如果【已知信息】里没有答案，就说不知道，不要瞎编。

【已知信息】：
{context}

用户问题：{question}
"""

prompt = ChatPromptTemplate.from_template(template)

# 4. 创建处理链 (Chain)
# 这就是 LangChain 的魔力：把 提示词模板 -> 模型 串联起来
chain = prompt | llm

# 5. 测试
question = "Thevilxx 的秘密武器是什么？"
print(f"\n[问]: {question}")
print("[AI 思考中]...")

response = chain.invoke({"context": secret_content, "question": question})

print(f"[答]: {response}")