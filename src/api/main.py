"""
API 入口 v2
新增：
- TracingMiddleware 自动注入 trace_id
- GET /api/v1/trace/{trace_id}
- GET /api/v1/trace/search
- GET /api/v1/metrics/rag
- POST /api/v1/skills/install
- POST /api/v1/skills/execute
- GET /api/v1/skills/list
"""
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import server_config, tracing_config, skills_config

app = FastAPI(
    title="companyrag - 企业知识库 RAG API",
    description="可观测 + 结构化 + 可扩展的企业级知识检索系统",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=server_config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Tracing middleware ----------
if tracing_config.ENABLED:
    from src.tracing import TracingMiddleware, get_tracer
    # tracer 单例已存在；middleware 会从 header 注入 trace_id
    app.add_middleware(TracingMiddleware)


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


class SkillInstallRequest(BaseModel):
    manifest: Dict[str, Any]
    entry_code: Optional[str] = None


class SkillExecuteRequest(BaseModel):
    name: str
    params: Dict[str, Any] = {}
    user_input: str = ""


# ==================== 健康检查 ====================

@app.get("/api/v1/health", summary="健康检查")
async def health_check():
    from src.storage.vector_store import get_vector_store
    store = get_vector_store()
    stats = store.get_stats() if store.has_collection() else {"row_count": 0}
    tracer = get_tracer() if tracing_config.ENABLED else None
    return {
        "status": "ok",
        "version": "0.2.0",
        "collection_exists": store.has_collection(),
        "total_chunks": stats.get("row_count", 0),
        "tracing_enabled": tracing_config.ENABLED,
        "skills_enabled": skills_config.ENABLED,
    }


# ==================== 检索 / 问答 ====================

@app.post("/api/v1/search", summary="向量检索")
async def search(request: SearchRequest):
    from src.retriever import get_retriever
    retriever = get_retriever()
    filter_expr = f'file_type == "{request.file_type}"' if request.file_type else None
    results = retriever.search(
        query=request.query, top_k=request.top_k,
        content_type=request.content_type, filter_expr=filter_expr,
    )
    return {"code": 0, "message": "success",
            "data": {"query": request.query, "total": len(results), "results": results}}


@app.post("/api/v1/chat", summary="RAG 问答")
async def chat(request: ChatRequest, x_trace_id: Optional[str] = None):
    from src.services.rag_service import get_rag_service
    from src.tracing.tracer import current_trace_id
    rag = get_rag_service()
    result = rag.chat(
        query=request.query, top_k=request.top_k,
        session_id=request.conversation_id,
        trace_id=current_trace_id() or x_trace_id,
    )
    result["trace_id"] = current_trace_id() or x_trace_id
    return {"code": 0, "message": "success", "data": result}


# ==================== 文档管理 ====================

@app.get("/api/v1/documents", summary="已索引文档列表")
async def list_documents():
    from src.services.index_service import get_index_service
    svc = get_index_service()
    return {"code": 0, "data": {"total": len(svc.list_indexed_files()), "items": svc.list_indexed_files()}}


@app.get("/api/v1/stats", summary="统计信息")
async def get_stats():
    from src.services.index_service import get_index_service
    svc = get_index_service()
    return {"code": 0, "data": {"total_chunks": svc.get_stats().get("row_count", 0),
                                 "files": svc.list_indexed_files()}}


# ==================== Tracing API ====================

@app.get("/api/v1/trace/{trace_id}", summary="查询一次完整链路")
async def get_trace(trace_id: str):
    if not tracing_config.ENABLED:
        raise HTTPException(400, "tracing disabled")
    tracer = get_tracer()
    events = tracer.get_trace(trace_id)
    return {"code": 0, "data": {"trace_id": trace_id, "events": events, "length": len(events)}}


@app.get("/api/v1/trace/search", summary="搜索 trace")
async def search_trace(q: str = "", limit: int = 50):
    if not tracing_config.ENABLED:
        raise HTTPException(400, "tracing disabled")
    tracer = get_tracer()
    return {"code": 0, "data": {"traces": tracer.search_traces(q, limit=limit)}}


@app.get("/api/v1/metrics/rag", summary="RAG 核心指标")
async def rag_metrics():
    if not tracing_config.ENABLED:
        raise HTTPException(400, "tracing disabled")
    return {"code": 0, "data": get_tracer().metrics()}


# ==================== Skills API ====================

def _get_runtime():
    from src.skills.registry import SkillRegistry
    from src.skills.runtime import SkillRuntime
    from src.services.rag_service import get_rag_service
    from src.services.index_service import get_index_service
    from src.storage.vector_store import get_vector_store
    from src.embedding.embedder import get_embedding_model
    from src.tracing.tracer import get_tracer

    registry = SkillRegistry(workspace_root=str(BASE_DIR))
    runtime = SkillRuntime(registry, deps={
        "rag_service": get_rag_service(),
        "index_service": get_index_service(),
        "vector_store": get_vector_store(),
        "embedder": get_embedding_model(),
        "tracer": get_tracer() if tracing_config.ENABLED else None,
        "workspace": str(BASE_DIR),
    })
    # 将 runtime 回注给 rag_service 让技能可被检索链调用
    get_rag_service().set_skills_runtime(runtime)
    return runtime


_skills_runtime = None


def get_skills_runtime():
    global _skills_runtime
    if _skills_runtime is None:
        _skills_runtime = _get_runtime()
    return _skills_runtime


@app.get("/api/v1/skills/list", summary="已安装 Skill 列表")
async def list_skills():
    if not skills_config.ENABLED:
        raise HTTPException(400, "skills disabled")
    return {"code": 0, "data": {"skills": get_skills_runtime().list_skills()}}


@app.post("/api/v1/skills/install", summary="语义/声明式安装 skill")
async def install_skill(req: SkillInstallRequest):
    if not skills_config.ENABLED:
        raise HTTPException(400, "skills disabled")
    sk = get_skills_runtime().install(req.manifest, entry_code=req.entry_code)
    return {"code": 0, "message": "installed", "data": sk.info()}


@app.post("/api/v1/skills/execute", summary="执行 skill")
async def execute_skill(req: SkillExecuteRequest, x_trace_id: Optional[str] = None):
    if not skills_config.ENABLED:
        raise HTTPException(400, "skills disabled")
    from src.tracing.tracer import current_trace_id
    rt = get_skills_runtime()
    res = rt.execute(
        name=req.name,
        params=req.params,
        user_input=req.user_input,
        trace_id=current_trace_id() or x_trace_id,
    )
    return {
        "code": 0 if res.ok else 1,
        "data": {
            "ok": res.ok,
            "output": res.output,
            "error": res.error,
            "took_ms": res.took_ms,
            "trace_id": current_trace_id() or x_trace_id,
        },
    }


@app.get("/api/v1/skills/search", summary="语义匹配可用 skill")
async def search_skills(q: str = "", top_k: int = 5):
    if not skills_config.ENABLED:
        raise HTTPException(400, "skills disabled")
    return {"code": 0, "data": {"matches": get_skills_runtime().semantic_match(q, top_k=top_k)}}


# ==================== 启动钩子 ====================

@app.on_event("startup")
async def startup_event():
    print("🚀 companyrag API 服务启动中 (v0.2.0)...")
    if tracing_config.ENABLED:
        print("🔍 Tracing enabled (backend=%s)" % tracing_config.BACKEND)
    if skills_config.ENABLED:
        rt = get_skills_runtime()
        print(f"🧩 Skills enabled ({len(rt.list_skills())} registered)")
    print("✅ 服务启动完成")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=server_config.HOST, port=server_config.PORT, reload=True)
