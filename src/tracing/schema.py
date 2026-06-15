"""
Trace 数据结构
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    # Query 生命周期
    QUERY_REQUEST = "query_request"
    QUERY_EMBEDDING = "query_embedding"
    VECTOR_RETRIEVAL = "vector_retrieval"
    BM25_RETRIEVAL = "bm25_retrieval"
    RERANK_RESULT = "rerank_result"
    FINAL_CONTEXT = "final_context"
    LLM_RESPONSE = "llm_response"
    HALLUCINATION_SCORE = "hallucination_score"
    QUERY_COMPLETE = "query_complete"

    # 文档生命周期
    DOCUMENT_INGEST = "document_ingest"
    DOCUMENT_PARSE = "document_parse"
    CHUNK_CREATE = "chunk_create"
    EMBEDDING_STORE = "embedding_store"

    # Skill
    SKILL_INSTALL = "skill_install"
    SKILL_EXECUTE = "skill_execute"


@dataclass
class TraceEvent:
    """单个 trace 事件"""
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: Optional[str] = None
    session_id: Optional[str] = None

    event_type: str = ""
    stage: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    # 负载：记录详细数据（query、检索topK、候选结果条数、分数等）
    data: Dict[str, Any] = field(default_factory=dict)
    # 标签/元信息
    tags: Dict[str, str] = field(default_factory=dict)
    # 错误信息
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def new_span_id() -> str:
    return uuid.uuid4().hex[:12]
