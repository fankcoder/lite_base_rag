"""
API 入口
"""
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import server_config

app = FastAPI(
    title="companyrag - 企业知识库 RAG API",
    description="企业知识库多模态 RAG 系统 API",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=server_config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 请求/响应模型 ====================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    content_type: Optional[str] = None
    file_type: Optional[str] = None


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    stream: bool = False
    conversation_id: Optional[str] = None


class DocumentInfo(BaseModel):
    file_name: str
    file_type: str
    file_hash: str
    chunk_count: int
    create_time: Optional[float] = None


# ==================== 健康检查 ====================

@app.get("/api/v1/health", summary="健康检查")
async def health_check():
    from src.storage.vector_store import get_vector_store

    store = get_vector_store()
    stats = store.get_stats() if store.has_collection() else {"row_count": 0}

    return {
        "status": "ok",
        "version": "0.1.0",
        "collection_exists": store.has_collection(),
        "total_chunks": stats.get("row_count", 0),
    }


# ==================== 检索接口 ====================

@app.post("/api/v1/search", summary="向量检索")
async def search(request: SearchRequest):
    """
    纯向量检索，不生成回答
    """
    from src.retriever import get_retriever

    retriever = get_retriever()

    filter_expr = None
    if request.file_type:
        filter_expr = f'file_type == "{request.file_type}"'

    results = retriever.search(
        query=request.query,
        top_k=request.top_k,
        content_type=request.content_type,
        filter_expr=filter_expr,
    )

    return {
        "code": 0,
        "message": "success",
        "data": {
            "query": request.query,
            "total": len(results),
            "results": results,
        }
    }


# ==================== 问答接口 ====================

@app.post("/api/v1/chat", summary="RAG 问答")
async def chat(request: ChatRequest):
    """
    RAG 问答（目前是骨架版本，返回检索结果）
    配置 LLM 后可生成完整回答
    """
    from src.services.rag_service import get_rag_service

    rag_service = get_rag_service()

    result = rag_service.chat(query=request.query, top_k=request.top_k)

    return {
        "code": 0,
        "message": "success",
        "data": result,
    }


# ==================== 文档管理接口 ====================

@app.get("/api/v1/documents", summary="获取已索引文档列表")
async def list_documents():
    from src.services.index_service import get_index_service

    service = get_index_service()
    files = service.list_indexed_files()

    return {
        "code": 0,
        "data": {
            "total": len(files),
            "items": files,
        }
    }


@app.get("/api/v1/stats", summary="获取统计信息")
async def get_stats():
    from src.services.index_service import get_index_service

    service = get_index_service()
    stats = service.get_stats()
    files = service.list_indexed_files()

    return {
        "code": 0,
        "data": {
            "total_documents": len(files),
            "total_chunks": stats.get("row_count", 0),
            "files": files,
        }
    }


# ==================== 启动钩子 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    print("🚀 companyrag API 服务启动中...")

    # 预加载模型（可选，首次请求时加载也可以）
    # from src.embedding import get_embedding_model
    # get_embedding_model()

    print("✅ 服务启动完成")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=server_config.HOST,
        port=server_config.PORT,
        reload=True,
    )
