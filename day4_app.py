import gradio as gr
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 第一部分：初始化 (Server Startup) ---
# 这部分代码只在程序启动时运行一次。
# 就像开饭馆，要在开门前把厨师（LLM）、菜谱（Prompt）和食材库（ChromaDB）都准备好。

print("系统启动中：正在加载模型和知识库...")

# 1. 准备模型
llm = OllamaLLM(model="qwen2.5:1.5b")
embeddings = OllamaEmbeddings(model="qwen2.5:1.5b")

# 2. 连接已有的向量数据库
# 注意：这里我们不需要再 TextLoader -> Split 了，因为 Day 3 已经存好了！
# 我们直接连接那个 "./chroma_db" 文件夹
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever()

# 3. 组装 RAG 链
template = """
你是一个高情商的 AI 助手。请根据下面的【机密信息】回答用户问题。
如果信息里没有提到，就礼貌地告诉用户你不知道，不要编造。

【机密信息】：
{context}

用户问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("系统就绪！")

# --- 第二部分：定义交互函数 (The Logic) ---
# Gradio 会把用户输入的 message 传给这个函数
# history 是之前的聊天记录（我们暂时不用，专注于回答当前问题）

def ask_ai(message, history):
    # 这里的 message 就是你在网页输入框里打的字
    response = rag_chain.invoke(message)
    return response

# --- 第三部分：搭建界面 (The UI) ---
# ChatInterface 是 Gradio 专门为聊天机器人提供的现成组件
demo = gr.ChatInterface(
    fn=ask_ai,
    title="Thevilxx 的私人知识库助手",
    description="基于 RAG 技术构建，你可以问我关于简历、秘密武器或鸿蒙开发的问题。",
    examples=["Thevilxx 的秘密武器是什么？", "鸿蒙开发用什么语言？", "你觉得 Thevilxx 能拿到实习吗？"]
)

# 启动网页
if __name__ == "__main__":
    demo.launch()