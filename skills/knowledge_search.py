"""
内置技能：知识库检索 & 问答
供 SkillRuntime 调用（entry: run(ctx) -> SkillResult）
"""
from __future__ import annotations

from typing import Any, Dict

from src.skills.skill import SkillContext, SkillResult


def run(ctx: SkillContext) -> SkillResult:
    query = ctx.user_input or (ctx.params or {}).get("query", "")
    top_k = int((ctx.params or {}).get("top_k", 5))

    if not query:
        return SkillResult(ok=False, error="缺少 query 参数")

    rag = ctx.rag_service
    if rag is None:
        return SkillResult(ok=False, error="RAGService 未注入")

    if ctx.tracer is not None:
        ctx.tracer.event("skill_execute", data={"skill": "knowledge-base-search", "query": query, "top_k": top_k})

    result: Dict[str, Any] = rag.chat(query, top_k=top_k)

    return SkillResult(
        ok=True,
        output=result,
    )
