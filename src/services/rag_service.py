"""
RAG 服务 - 整合检索和生成
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.retriever import get_retriever
from config import retriever_config


class RAGService:
    """RAG 问答服务"""

    def __init__(self):
        self.retriever = get_retriever()

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        检索相关文档
        """
        top_k = top_k or retriever_config.TOP_K
        return self.retriever.search(query, top_k=top_k)

    def generate(self, query: str, contexts: List[Dict]) -> str:
        """
        基于上下文生成回答（需要配置 LLM）
        这里是骨架，具体实现见 src/generator/
        """
        # 构造 prompt
        context_text = self._format_contexts(contexts)

        system_prompt = f"""你是企业知识库问答助手，必须严格基于提供的参考资料回答用户问题。

【重要规则】
1. 只能使用参考资料中的信息回答问题。
2. 如果参考资料中没有答案，直接说"抱歉，知识库中没有找到相关信息"，不要编造。
3. 答案末尾必须标注来源。
4. 对于数字、日期、人名等精确信息，必须确保和原文一致。
5. 不要做过度推断，只说资料中明确提到的内容。

【参考资料】
{context_text}

请根据以上参考资料回答用户问题。"""

        # 这里只是骨架，实际调用 LLM 需要配置 API
        # 详见 src/generator/llm_client.py
        return f"[RAG 骨架] 问题: {query}\n\n检索到 {len(contexts)} 条相关文档，请配置 LLM 后生成回答。"

    async def generate_stream(self, query: str, contexts: List[Dict]) -> AsyncGenerator[str, None]:
        """流式生成（骨架）"""
        answer = self.generate(query, contexts)
        for char in answer:
            yield char

    def chat(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        完整 RAG 问答
        """
        # 1. 检索
        contexts = self.retrieve(query, top_k=top_k)

        if not contexts:
            return {
                "query": query,
                "answer": "抱歉，知识库中没有找到相关信息。",
                "sources": [],
                "contexts": [],
            }

        # 2. 生成
        answer = self.generate(query, contexts)

        # 3. 整理来源
        sources = self._extract_sources(contexts)

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
        }

    def _format_contexts(self, contexts: List[Dict]) -> str:
        """将检索结果格式化为 prompt 上下文"""
        parts = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get('file_name', '未知来源')
            if ctx.get('page_num'):
                source += f" (第 {ctx['page_num']} 页)"
            if ctx.get('slide_num'):
                source += f" (PPT第 {ctx['slide_num']} 页)"

            content = ctx.get('content', '')
            # 截取关键内容，避免 prompt 过长
            if len(content) > 1000:
                content = content[:1000] + "..."

            parts.append(f"[{i}] {source}\n{content}")

        return "\n\n".join(parts)

    def _extract_sources(self, contexts: List[Dict]) -> List[Dict]:
        """提取来源信息"""
        sources = []
        for ctx in contexts:
            sources.append({
                "file_name": ctx.get("file_name", ""),
                "file_type": ctx.get("file_type", ""),
                "page_num": ctx.get("page_num", 0),
                "slide_num": ctx.get("slide_num", 0),
                "chapter_path": ctx.get("chapter_path", ""),
                "content": ctx.get("content", "")[:200],  # 预览
                "score": ctx.get("score", 0),
            })
        return sources


# 全局单例
_rag_service = None


def get_rag_service() -> RAGService:
    """获取 RAG 服务单例"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
