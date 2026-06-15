"""
Trace 存储抽象层
- 默认 InMemory（适合调试/单进程）
- 可替换为 Elasticsearch / ClickHouse / Loki 等，只需实现 TraceStore 接口
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .schema import TraceEvent


class TraceStore(ABC):
    """Trace 存储抽象接口，后续可直接扩展为 ES / CK / Loki adapter"""

    @abstractmethod
    def append(self, event: TraceEvent) -> None: ...

    @abstractmethod
    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def metrics(self) -> Dict[str, Any]: ...


class InMemoryTraceStore(TraceStore):
    """
    内存环形缓冲 trace 存储
    - 默认保留最近 N 条 trace
    - 线程安全
    """

    def __init__(self, max_events: int = 100_000):
        self._events: deque = deque(maxlen=max_events)
        self._lock = threading.Lock()
        # trace_id -> events list 索引
        self._trace_index: Dict[str, List[Dict[str, Any]]] = {}

    def append(self, event: TraceEvent) -> None:
        d = event.to_dict()
        with self._lock:
            self._events.append(d)
            bucket = self._trace_index.setdefault(event.trace_id, [])
            bucket.append(d)
            # 避免索引无限增长
            if len(self._trace_index) > 5000:
                # 删除最老的一个 trace
                oldest_tid = next(iter(self._trace_index))
                self._trace_index.pop(oldest_tid, None)

    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._trace_index.get(trace_id, []))

    def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        按 trace_id/query text/文件名字符串包含进行简单搜索
        生产环境可替换为 ES full-text search
        """
        q = (query or "").strip().lower()
        hits: List[Dict[str, Any]] = []
        seen_ids = set()
        with self._lock:
            # 反向遍历优先返回新 trace
            for ev in reversed(list(self._events)):
                if len(hits) >= limit:
                    break
                tid = ev.get("trace_id")
                if tid in seen_ids:
                    continue
                blob = str(ev).lower()
                if q and q not in blob:
                    continue
                seen_ids.add(tid)
                hits.append({
                    "trace_id": tid,
                    "session_id": ev.get("session_id"),
                    "event_type": ev.get("event_type"),
                    "timestamp": ev.get("timestamp"),
                    "preview": (ev.get("data") or {}).get("query")
                        or (ev.get("data") or {}).get("file_name")
                        or "",
                })
        return hits

    def metrics(self) -> Dict[str, Any]:
        """聚合 RAG 核心指标"""
        now = time.time()
        last_1h = now - 3600
        last_5m = now - 300

        q_total = q_1h = q_5m = 0
        ingest_total = ingest_1h = ingest_5m = 0
        latencies: List[float] = []
        scores: List[float] = []

        with self._lock:
            for ev in self._events:
                et = ev.get("event_type")
                ts = ev.get("timestamp", 0)
                if et == "query_request":
                    q_total += 1
                    if ts >= last_1h:
                        q_1h += 1
                    if ts >= last_5m:
                        q_5m += 1
                elif et == "document_ingest":
                    ingest_total += 1
                    if ts >= last_1h:
                        ingest_1h += 1
                    if ts >= last_5m:
                        ingest_5m += 1
                elif et == "query_complete":
                    latencies.append(ev.get("duration_ms", 0) or 0)
                elif et == "rerank_result":
                    top = (ev.get("data") or {}).get("top_score")
                    if top is not None:
                        scores.append(float(top))

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95_latency = _percentile(latencies, 95) if latencies else 0.0
        avg_top_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "queries_total": q_total,
            "queries_last_1h": q_1h,
            "queries_last_5m": q_5m,
            "ingests_total": ingest_total,
            "ingests_last_1h": ingest_1h,
            "ingests_last_5m": ingest_5m,
            "avg_query_latency_ms": round(avg_latency, 2),
            "p95_query_latency_ms": round(p95_latency, 2),
            "avg_top_retrieval_score": round(avg_top_score, 4),
            "total_events_stored": len(self._events),
        }


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)
