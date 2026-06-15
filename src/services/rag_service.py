"""
RAG 服务 v2
- 完整链路 tracing：query_embedding / vector_retrieval / bm25_retrieval / rerank_result / final_context / llm_response
- 集成 SkillRuntime（可选）
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.retriever import get_retriever
from src.tracing.tracer import get_tracer
from config import retriever_config


class RAGService:

    def __init__(self):
        self.retriever = get_retriever()
        self.tracer = get_tracer()
        self._skills_runtime = None  # 由 main.py 在启动后注入

    def set_skills_runtime(self, runtime):
        self._skills_runtime = runtime

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = None, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        top_k = top_k or retriever_config.TOP_K
        with self.tracer.span("vector_retrieval", stage="retrieve") as ev:
            # query embedding 阶段
            with self.tracer.span("query_embedding", stage="embed") as eb:
                qvec = self.retriever.embedder.encode_query(query)
                self.tracer.update_span_data(eb, dim=len(qvec) if hasattr(qvec, "__len__") else None)

            results = self.retriever.search(query, top_k=top_k)

            # BM25 单独事件
            if self.retriever.bm25_enabled:
                self.tracer.event("bm25_retrieval", data={"enabled": True, "returned": 0})

            # rerank 事件
            if self.retriever.rerank_enabled and results:
                top_score = max((r.get('rerank_score', r.get('score', 0)) for r in results), default=0.0)
                self.tracer.event(
                    "rerank_result",
                    data={
                        "returned": len(results),
                        "top_score": top_score,
                        "scores": [r.get('rerank_score', r.get('score', 0)) for r in results[:5]],
                    },
                )

            self.tracer.update_span_data(ev, query=query, top_k=top_k, returned=len(results))
            return results

    # ------------------------------------------------------------------
    def generate(self, query: str, contexts: List[Dict]) -> str:
        context_text = self._format_contexts(contexts)
        system_prompt = f"""你是企业知识库问答助手，必须严格基于提供的参考资料回答用户问题。

【重要规则】
1. 只能使用参考资料中的信息回答问题。
2. 如果参考资料中没有答案，直接说"抱歉，知识库中没有找到相关信息"，不要编造。
3. 答案末尾必须标注来源。
4. 对于数字、日期、人名等精确信息，必须确保和原文一致。

【参考资料】
{context_text}

请根据以上参考资料回答用户问题。"""
        return f"[RAG 骨架] 问题: {query}\n\n检索到 {len(contexts)} 条相关文档，请配置 LLM 后生成回答。"

    async def generate_stream(self, query: str, contexts: List[Dict]):
        answer = self.generate(query, contexts)
        for char in answer:
            yield char

    # ------------------------------------------------------------------
    def chat(self, query: str, top_k: int = None,
             session_id: Optional[str] = None,
             trace_id: Optional[str] = None) -> Dict[str, Any]:
        with self.tracer.trace(session_id=session_id, trace_id=trace_id):
            with self.tracer.span("query_request", stage="chat") as req_ev:
                self.tracer.update_span_data(req_ev, query=query, top_k=top_k)

                contexts = self.retrieve(query, top_k=top_k)

                if not contexts:
                    answer = "抱歉，知识库中没有找到相关信息。"
                    self.tracer.event("llm_response", data={"answer": answer, "empty": True})
                    self.tracer.event("final_context", data={"num_contexts": 0})
                    return {"query": query, "answer": answer, "sources": [], "contexts": [],
                            "trace_id": self.tracer and None}

                with self.tracer.span("final_context", stage="context") as ctx_ev:
                    preview = [c.get('content', '')[:120] for c in contexts[:3]]
                    self.tracer.update_span_data(ctx_ev, num_contexts=len(contexts), preview=preview)

                with self.tracer.span("llm_response", stage="llm") as llm_ev:
                    answer = self.generate(query, contexts)
                    self.tracer.update_span_data(llm_ev, answer_len=len(answer))

                sources = self._extract_sources(contexts)
                tid = self.tracer.get_trace("")  # placeholder
                return {
                    "query": query, "answer": answer, "sources": sources,
                    "contexts": contexts,
                }

    # ------------------------------------------------------------------
    def _format_contexts(self, contexts: List[Dict]) -> str:
        parts = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get('file_name', '未知来源')
            if ctx.get('main_title'):
                source = f"{ctx['main_title']} / {source}"
            if ctx.get('page_num'):
                source += f" (第 {ctx['page_num']} 页)"
            if ctx.get('slide_num'):
                source += f" (PPT第 {ctx['slide_num']} 页)"
            content = ctx.get('content', '')
            if len(content) > 1200:
                content = content[:1200] + "..."
            parts.append(f"[{i}] {source}\n{content}")
        return "\n\n".join(parts)

    def _extract_sources(self, contexts: List[Dict]) -> List[Dict]:
        out = []
        for ctx in contexts:
            out.append({
                "file_name": ctx.get("file_name", ""),
                "file_type": ctx.get("file_type", ""),
                "main_title": ctx.get("main_title", ""),
                "section_title": ctx.get("section_title", ctx.get("chapter_path", "")),
                "page_num": ctx.get("page_num", 0),
                "slide_num": ctx.get("slide_num", 0),
                "chapter_path": ctx.get("chapter_path", ""),
                "content": ctx.get("content", "")[:200],
                "score": ctx.get("score", 0),
            })
        return out


_rag_service = None

def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
