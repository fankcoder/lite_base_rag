"""
Skill 数据结构 + Skill 实例
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SkillManifest:
    """声明式 manifest"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    entry: str = ""
    capabilities: List[str] = field(default_factory=list)
    embedding_required: bool = False
    inputs: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=lambda: ["local_read"])
    author: str = ""
    tags: List[str] = field(default_factory=list)
    icon: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillManifest":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class SkillContext:
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    rag_service: Any = None
    index_service: Any = None
    vector_store: Any = None
    embedder: Any = None
    tracer: Any = None
    user_input: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    workspace: str = ""


@dataclass
class SkillResult:
    ok: bool = True
    output: Any = None
    error: Optional[str] = None
    took_ms: float = 0.0
    traces: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Skill:
    """可执行的 Skill 实例"""
    manifest: SkillManifest
    skill_dir: str = ""
    entry_path: str = ""
    module: Any = None
    run_fn: Optional[Callable[..., Any]] = None

    def run(self, ctx: SkillContext) -> SkillResult:
        start = time.time()
        try:
            if self.run_fn is None:
                return SkillResult(
                    ok=False,
                    error=f"skill {self.manifest.name} 的 entry 未实现 run(ctx)",
                    took_ms=(time.time() - start) * 1000.0,
                )
            raw = self.run_fn(ctx)
            if isinstance(raw, SkillResult):
                raw.took_ms = (time.time() - start) * 1000.0
                return raw
            return SkillResult(ok=True, output=raw, took_ms=(time.time() - start) * 1000.0)
        except Exception as e:
            return SkillResult(ok=False, error=str(e), took_ms=(time.time() - start) * 1000.0)

    def safe_run(self, ctx: SkillContext) -> SkillResult:
        """可扩展：权限守卫 / 超时控制 / 资源限制"""
        return self.run(ctx)

    def info(self) -> dict:
        return {
            "name": self.manifest.name,
            "version": self.manifest.version,
            "description": self.manifest.description,
            "capabilities": self.manifest.capabilities,
            "entry": self.entry_path,
            "permissions": self.manifest.permissions,
            "embedding_required": self.manifest.embedding_required,
        }
