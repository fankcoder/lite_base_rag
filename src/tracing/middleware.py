"""
FastAPI Middleware - 为每个入站请求自动注入 trace_id / session_id
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .tracer import _session_id_var, _trace_id_var, get_tracer


class TracingMiddleware(BaseHTTPMiddleware):
    """
    1. 从 header 读取 x-trace-id / x-session-id（没有则生成）
    2. 写入 contextvars，后续代码可通过 tracer.trace() 直接使用
    3. 请求结束时记录 query_request / query_complete 或 generic http request 事件
    """

    async def dispatch(self, request: Request, call_next):
        trace_id: Optional[str] = request.headers.get("x-trace-id") or uuid.uuid4().hex[:16]
        session_id: Optional[str] = (
            request.headers.get("x-session-id")
            or request.headers.get("x-conversation-id")
        )

        t_tok = _trace_id_var.set(trace_id)
        s_tok = _session_id_var.set(session_id)

        start = time.time()
        tracer = get_tracer()
        try:
            # 记录入口
            tracer.event(
                "query_request" if request.url.path.startswith(("/api/v1/chat", "/api/v1/search")) else "http_request",
                data={
                    "method": request.method,
                    "path": request.url.path,
                    "query": request.url.query,
                    "client": request.client.host if request.client else "",
                },
                tags={"endpoint": request.url.path},
            )
            response: Response = await call_next(request)
            duration_ms = (time.time() - start) * 1000.0
            response.headers["x-trace-id"] = trace_id
            if session_id:
                response.headers["x-session-id"] = session_id
            tracer.event(
                "query_complete" if request.url.path.startswith(("/api/v1/chat", "/api/v1/search")) else "http_complete",
                data={"status_code": response.status_code, "duration_ms": round(duration_ms, 2)},
                duration_ms=duration_ms,
            )
            return response
        except Exception as e:
            duration_ms = (time.time() - start) * 1000.0
            tracer.event(
                "query_complete",
                data={"error": str(e)},
                error=str(e),
                duration_ms=duration_ms,
            )
            raise
        finally:
            _trace_id_var.reset(t_tok)
            _session_id_var.reset(s_tok)
