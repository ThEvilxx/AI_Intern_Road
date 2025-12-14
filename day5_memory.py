import gradio as gr
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 1. 初始化 (和之前一样)
print("正在初始化大脑区域...")
llm = OllamaLLM(model="qwen2.5:1.5b")
embeddings = OllamaEmbeddings(model="qwen2.5:1.5b")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# --- 核心升级：历史问题改写器 ---
# 这里的任务是：如果用户的问题依赖上下文（比如说了"它"），就把它改写成独立问题。
contextualize_q_system_prompt = """
给定一段聊天记录和一个最新的用户问题，
如果这个问题引用了上下文（比如使用了"它"、"这个"），
请把它改写成一个独立的问题，使其不需要上下文也能被理解。
不要回答问题，只负责改写问题。如果不需要改写，原样输出。
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 这个链条的作用：(历史 + 新问题) -> 改写后的问题
history_aware_retriever = contextualize_q_prompt | llm | StrOutputParser()

# --- 问答链 (QA Chain) ---
qa_system_prompt = """
你是一个助手。请根据下面的上下文回答问题。
如果不知道，就诚实地说不知道。

上下文：
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"), # 把历史也塞进回答的 Prompt 里
    ("human", "{input}"),
])

# 定义 RAG 逻辑：
# 1. 先用 retrieve 找资料（注意：这里我们还没把改写器串联进去，为了简化，我们在函数里手动串联）
# LangChain 的完整写法比较复杂，为了让你看懂，我们用“函数式”写法

def chat_logic(message, history):
    # history 是 Gradio 传过来的列表：[['用户问1', 'AI答1'], ['用户问2', 'AI答2']]
    # 我们要把它转换成 LangChain 认识的格式
    langchain_history = []
    for human_msg, ai_msg in history:
        langchain_history.append(HumanMessage(content=human_msg))
        langchain_history.append(AIMessage(content=ai_msg))
    
    # 步骤 1: 改写问题 (解决"它"的问题)
    # 只有当有历史记录时才需要改写
    if langchain_history:
        print(f"👀 正在分析历史上下文...")
        reformulated_question = history_aware_retriever.invoke({
            "chat_history": langchain_history,
            "input": message
        })
        print(f"🔄 问题已改写为: {reformulated_question}")
    else:
        reformulated_question = message

    # 步骤 2: 拿着改写后的问题去检索
    docs = retriever.invoke(reformulated_question)
    context_text = "\n\n".join([d.page_content for d in docs])

    # 步骤 3: 生成回答
    # 我们直接把 context 和 history 塞给 LLM
    final_prompt = qa_prompt.format(
        context=context_text,
        chat_history=langchain_history,
        input=message
    )
    
    response = llm.invoke(final_prompt)
    return response

# --- 启动界面 ---
print("全功能 AI 助手已就绪！")
gr.ChatInterface(
    fn=chat_logic,
    title="Thevilxx 的全功能 RAG (带记忆版)",
    description="我已经治好了失忆症，你可以尝试问我：'显卡是什么？' 然后追问 '它有多少显存？'",
    theme="ocean"
).launch()