"""
Tracer 核心 - 创建 span、记录事件、异步写入存储
使用 contextvars 保证异步上下文内 trace_id/span_id 自动传播
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from .schema import EventType, TraceEvent, new_span_id, new_trace_id
from .store import InMemoryTraceStore, TraceStore


logger = logging.getLogger("rag.tracer")

# 上下文变量
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("trace_id", default=None)
_session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("session_id", default=None)
_parent_span_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("parent_span", default=None)


class Tracer:
    """
    RAG 链路追踪器
    - 生成/传播 trace_id、session_id、span_id
    - 事件通过后台线程异步写入 TraceStore，不阻塞主路径
    """

    def __init__(self, store: Optional[TraceStore] = None, max_workers: int = 2):
        self.store: TraceStore = store or InMemoryTraceStore()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="trace-writer")

    # ---------- 上下文管理 ----------

    @contextmanager
    def trace(self, session_id: Optional[str] = None, trace_id: Optional[str] = None) -> Iterator[str]:
        """开启一次新的 trace（query 级别），yield trace_id"""
        tid = trace_id or new_trace_id()
        sid = session_id or _session_id_var.get()

        t_tok = _trace_id_var.set(tid)
        s_tok = _session_id_var.set(sid)
        p_tok = _parent_span_var.set(None)
        try:
            yield tid
        finally:
            _trace_id_var.reset(t_tok)
            _session_id_var.reset(s_tok)
            _parent_span_var.reset(p_tok)

    @contextmanager
    def span(self, event_type: str, stage: str = "",
             extra_tags: Optional[Dict[str, str]] = None) -> Iterator[TraceEvent]:
        """
        一个计时 span：进入时记录 parent_id，退出时自动记录 duration_ms
        """
        tid = _trace_id_var.get() or new_trace_id()
        # 如果没有 trace，自动创建一个
        auto_created = _trace_id_var.get() is None
        if auto_created:
            _trace_id_var.set(tid)

        sid = _session_id_var.get()
        parent = _parent_span_var.get()
        span_id = new_span_id()
        parent_tok = _parent_span_var.set(span_id)

        start = time.time()
        ev = TraceEvent(
            trace_id=tid,
            span_id=span_id,
            parent_id=parent,
            session_id=sid,
            event_type=event_type,
            stage=stage or event_type,
            tags=dict(extra_tags or {}),
        )
        try:
            yield ev
            ev.duration_ms = (time.time() - start) * 1000.0
            self.emit(ev)
        except Exception as e:
            ev.duration_ms = (time.time() - start) * 1000.0
            ev.error = str(e)
            self.emit(ev)
            raise
        finally:
            _parent_span_var.reset(parent_tok)
            if auto_created:
                _trace_id_var.set(None)

    # ---------- 事件写入 ----------

    def event(self, event_type: str, data: Optional[Dict[str, Any]] = None,
              tags: Optional[Dict[str, str]] = None,
              stage: str = "", error: Optional[str] = None,
              duration_ms: Optional[float] = None) -> TraceEvent:
        """一次性记录事件（非 span 场景）"""
        tid = _trace_id_var.get() or new_trace_id()
        parent = _parent_span_var.get()
        ev = TraceEvent(
            trace_id=tid,
            span_id=new_span_id(),
            parent_id=parent,
            session_id=_session_id_var.get(),
            event_type=event_type,
            stage=stage or event_type,
            data=dict(data or {}),
            tags=dict(tags or {}),
            error=error,
            duration_ms=duration_ms or 0.0,
        )
        self.emit(ev)
        return ev

    def update_span_data(self, ev: TraceEvent, **kwargs) -> None:
        """向正在进行的 span 追加 data 字段"""
        ev.data.update(kwargs)

    def emit(self, ev: TraceEvent) -> None:
        """异步写入 store，不阻塞主链路"""
        try:
            self._executor.submit(self._safe_append, ev)
        except Exception as e:
            logger.warning("trace submit failed: %s", e)

    def _safe_append(self, ev: TraceEvent) -> None:
        try:
            # 同时输出一条 JSON 日志，方便接入 Loki / ES
            logger.info("RAG_TRACE %s", json.dumps(ev.to_dict(), ensure_ascii=False, default=str))
            self.store.append(ev)
        except Exception as e:
            logger.warning("trace store append failed: %s", e)

    # ---------- 查询 ----------

    def get_trace(self, trace_id: str):
        return self.store.get_trace(trace_id)

    def search_traces(self, query: str, limit: int = 50):
        return self.store.search(query, limit=limit)

    def metrics(self):
        return self.store.metrics()


# ---------- 全局单例 ----------

_global_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def set_global_tracer(tracer: Tracer) -> None:
    global _global_tracer
    _global_tracer = tracer


def current_trace_id() -> Optional[str]:
    return _trace_id_var.get()


def current_session_id() -> Optional[str]:
    return _session_id_var.get()


def set_session_id(sid: str) -> None:
    _session_id_var.set(sid)
