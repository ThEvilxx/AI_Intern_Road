import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 准备大模型和嵌入模型
# OllamaEmbeddings 负责把文字变成数字向量，我们复用 qwen2.5:1.5b (它也能做简单的embedding，虽然不专业但够用)
# *注意：专业生产环境会用 bge-m3 等专用 embedding 模型，这里为了省事直接复用*
print("正在初始化模型组件...")
llm = OllamaLLM(model="qwen2.5:1.5b")
embeddings = OllamaEmbeddings(model="qwen2.5:1.5b")

# 2. 加载并切分文档
# 为什么切分？因为大模型吃不下整本书，只能吃小块
print("正在处理文档...")
loader = TextLoader("./data/secret.txt", encoding='utf-8')
docs = loader.load()

# 切分器：每块 100 个字符，重叠 20 个字符（防止切断关键句）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(docs)

# 3. 建立向量数据库 (这是今天的核心！)
# 这一步会把文字变成向量，存到本地的一个文件夹里
print("正在构建向量索引 (可能会卡顿一下)...")
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings,
    persist_directory="./chroma_db" # 数据持久化存储在这里
)
retriever = vectorstore.as_retriever()

# 4. 构建 Prompt 模板
template = """
你是一个智能助手。请根据下面的【相关片段】回答问题。
如果片段里没有答案，就说不知道。

【相关片段】：
{context}

用户问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 5. 构建 RAG 链
# 这里的逻辑是：问题 -> 去数据库搜(retriever) -> 搜到的内容填入 prompt -> 给 LLM -> 输出字符串
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. 测试提问
# 问一个昨天能回的，再问一个今天新增的
questions = ["Thevilxx 的秘密武器是什么？", "鸿蒙开发的主力语言是什么？"]

print("-" * 30)
for q in questions:
    print(f"\n[问]: {q}")
    print(f"[AI 检索并回答中]...")
    response = rag_chain.invoke(q)
    print(f"[答]: {response}")
print("-" * 30)