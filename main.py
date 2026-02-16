"""
RAG 系统后端 - 生态学文献知识库
使用 langchain-ollama (官方新版) 适配 Qwen 2.5 32B
"""
import os
import json
import asyncio
import aiofiles
import pypdf
from typing import List, Optional

# FastAPI 相关
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- 核心改动：使用新版库 ---
from langchain_core.prompts import PromptTemplate        # 解决 PromptTemplate 找不到
from langchain_text_splitters import RecursiveCharacterTextSplitter # 解决 text_splitter 找不到
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # 新版 Ollama
from langchain_chroma import Chroma                       # 新版 Chroma

# 配置
UPLOAD_DIR = "data/documents"
CHROMA_DIR = "data/chroma_db"
FRONTEND_DIR = "../frontend"

# 模型配置 (32B 模型较大，建议保持适当的上下文窗口)
MODEL_NAME = "qwen2.5:32b-instruct"
EMBEDDING_MODEL = "nomic-embed-text"

# 全局变量
vectorstore = None
embeddings = None
llm = None

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

def load_vectorstore():
    """加载向量数据库"""
    global vectorstore, embeddings
    try:
        vectorstore = Chroma(
            collection_name="ecology_docs",
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        print("向量数据库加载成功")
    except Exception as e:
        print(f"向量数据库加载警告: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global embeddings, llm
    print(">>> 正在初始化系统 (LangChain-Ollama 新版)...")
    
    # 1. 初始化 Embedding
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 2. 初始化 LLM (OllamaLLM)
    # 新版库直接支持 num_ctx 等参数，无需嵌套在 kwargs 里
    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=0.2,
        num_ctx=32768,       # 上下文窗口
        num_predict=4096,    # 最大生成长度
        top_p=0.9,
        repeat_penalty=1.2
    )

    # 3. 加载数据库
    load_vectorstore()
    
    print(">>> 系统启动完成！")
    yield
    print(">>> 系统正在关闭...")

app = FastAPI(title="生态学文献 RAG 系统", lifespan=lifespan)

# CORS 设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
)

# Prompt 模板
PROMPT_TEMPLATE = """你是一位资深的生态学教授和研究助理。
所有问题的回答均以人类工作者的语言习惯，加强表达清晰度、逻辑结构、科学准确性与语法。
在有上下文的情况下，请详细阅读【上下文信息】来回答【问题】。
如果你不知道答案，请直接说"我不知道"，不要编造信息。

上下文信息：
{context}

问题：{question}

详细回答："""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# 数据模型
class QueryRequest(BaseModel):
    question: str
    top_k: int = 4
    message_id: Optional[int] = None # 支持前端传来的消息ID

# --- 路由定义 ---

@app.get("/")
async def root():
    """返回前端页面"""
    index_path = os.path.join(FRONTEND_DIR, "templates", "index.html")
    # 兼容性检查：如果 templates 目录不存在，尝试直接找 index.html
    if not os.path.exists(index_path):
        fallback_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(fallback_path):
            return FileResponse(fallback_path)
        raise HTTPException(status_code=404, detail="前端 index.html 未找到")
    return FileResponse(index_path)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传并处理文档"""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # 提取文本
        text = ""
        if file.filename.lower().endswith('.pdf'):
            try:
                with open(file_path, 'rb') as pdf_file:
                    reader = pypdf.PdfReader(pdf_file)
                    for page in reader.pages:
                        text += page.extract_text()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"PDF 解析失败: {e}")
        elif file.filename.lower().endswith('.txt'):
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                text = await f.read()
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        # 分割并入库
        chunks = text_splitter.split_text(text)
        if vectorstore:
            # 构建 metadata
            metadatas = [{"source": file.filename, "chunk": i} for i in range(len(chunks))]
            await vectorstore.aadd_texts(texts=chunks, metadatas=metadatas)
        
        return JSONResponse({"message": "上传成功", "filename": file.filename})
    except Exception as e:
        print(f"上传错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(request: QueryRequest):
    """流式查询接口 - 统一 Prompt 风格版"""
    
    # 检查数据库状态
    has_docs = False
    try:
        if vectorstore:
            has_docs = True 
    except:
        has_docs = False

    async def generate_stream():
        try:
            sources = []
            context_text = "（当前未检索到特定参考文献，请根据您的资深学术知识储备进行回答）" # 默认提示词

            # 1. RAG 检索阶段
            if has_docs:
                try:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": request.top_k})
                    docs = await retriever.ainvoke(request.question)
                    
                    if docs:
                        # 提取来源
                        for doc in docs:
                            if doc.metadata and "source" in doc.metadata:
                                src = doc.metadata["source"]
                                if src not in sources:
                                    sources.append(src)
                        
                        # 发送来源给前端
                        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"

                        # 构造上下文
                        context_text = "\n\n".join([d.page_content for d in docs])
                except Exception as e:
                    print(f"检索出错: {e}")

            # 2. 统一构造 Full Prompt (无论是否有文档，都走 PROMPT 模板)
            full_prompt = PROMPT.format(context=context_text, question=request.question)

            # 3. LLM 流式生成
            if not llm:
                yield f"data: {json.dumps({'type': 'error', 'data': '模型未初始化'}, ensure_ascii=False)}\n\n"
                return

            async for chunk in llm.astream(full_prompt):
                if chunk:
                    yield f"data: {json.dumps({'type': 'text', 'data': chunk}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_msg = f"生成错误: {str(e)}"
            print(error_msg)
            yield f"data: {json.dumps({'type': 'error', 'data': error_msg}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

@app.get("/stats")
async def get_stats():
    """获取统计信息 - 实时统计片段数量"""
    try:
        # 1. 统计文件数量
        files = [f for f in os.listdir(UPLOAD_DIR) if not f.startswith('.')]
        
        # 2. 统计向量片段数量
        chunk_count = 0
        if vectorstore is not None:
            try:
                # 通过 vectorstore 访问底层的 collection 计数
                chunk_count = vectorstore._collection.count()
            except Exception as e:
                print(f"获取片段计数失败: {e}")
                chunk_count = 0
        
        return JSONResponse({
            "total_files": len(files),
            "files": files,
            "total_chunks": chunk_count  # 这里不再是 "N/A"，而是实际的数字
        })
    except Exception as e:
        print(f"获取状态异常: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/clear")
async def clear_database():
    """清空知识库"""
    try:
        global vectorstore
        import shutil
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        
        # 重新加载
        load_vectorstore()
        
        # 清空文件
        if os.path.exists(UPLOAD_DIR):
            for f in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        return JSONResponse({"message": "知识库已清空"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000)