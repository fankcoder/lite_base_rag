"""
Skill Runtime - 对外统一执行入口，打通 RAG/Index/Embedder/Tracer 依赖
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .registry import SkillRegistry
from .skill import SkillContext, SkillResult

logger = logging.getLogger("rag.skills.runtime")


class SkillRuntime:
    def __init__(self, registry: SkillRegistry, deps: Optional[Dict[str, Any]] = None):
        self.registry = registry
        self.deps: Dict[str, Any] = deps or {}

    def set_deps(self, **deps):
        self.deps.update(deps)

    def list_skills(self):
        return self.registry.list()

    def execute(self, name: str, params: Optional[Dict[str, Any]] = None,
                user_input: str = "",
                trace_id: Optional[str] = None,
                session_id: Optional[str] = None) -> SkillResult:
        skill = self.registry.get(name)
        if skill is None:
            return SkillResult(ok=False, error=f"skill 未安装: {name}")

        ctx = SkillContext(
            trace_id=trace_id,
            session_id=session_id,
            rag_service=self.deps.get("rag_service"),
            index_service=self.deps.get("index_service"),
            vector_store=self.deps.get("vector_store"),
            embedder=self.deps.get("embedder"),
            tracer=self.deps.get("tracer"),
            user_input=user_input,
            params=params or {},
            workspace=self.deps.get("workspace", ""),
        )

        tracer = self.deps.get("tracer")
        if tracer is not None:
            with tracer.span("skill_execute", stage=f"skill:{name}") as ev:
                tracer.update_span_data(ev, skill=name, params=params or {})
                result = skill.safe_run(ctx)
                tracer.update_span_data(
                    ev, ok=result.ok, took_ms=result.took_ms,
                    error=result.error,
                )
                return result
        return skill.safe_run(ctx)

    def install(self, manifest: dict, entry_code: Optional[str] = None):
        tracer = self.deps.get("tracer")
        if tracer is not None:
            tracer.event("skill_install", data={"name": manifest.get("name")})
        return self.registry.install_from_manifest(manifest, entry_code=entry_code)

    def semantic_match(self, query: str, top_k: int = 3):
        return self.registry.semantic_search(query, top_k=top_k)
