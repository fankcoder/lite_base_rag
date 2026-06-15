"""
RAG Tracing & Observability
为每一次 query / document 生命周期生成 trace_id，异步记录结构化事件
"""
from .tracer import (
    Tracer,
    TraceEvent,
    get_tracer,
    set_global_tracer,
)
from .middleware import TracingMiddleware
from .store import TraceStore, InMemoryTraceStore

__all__ = [
    "Tracer",
    "TraceEvent",
    "get_tracer",
    "set_global_tracer",
    "TracingMiddleware",
    "TraceStore",
    "InMemoryTraceStore",
]
