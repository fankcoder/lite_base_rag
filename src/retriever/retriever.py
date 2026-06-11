"""
检索模块 - 向量检索 + BM25 + Rerank
"""
import sys
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import retriever_config
from src.storage.vector_store import get_vector_store
from src.embedding.embedder import get_embedding_model


class Retriever:
    """多路检索器"""

    def __init__(self):
        self.vector_store = get_vector_store()
        self.embedder = get_embedding_model()
        self.top_k = retriever_config.TOP_K
        self.bm25_enabled = retriever_config.BM25_ENABLED
        self.rerank_enabled = retriever_config.RERANK_ENABLED
        self.similarity_threshold = retriever_config.SIMILARITY_THRESHOLD

        self._bm25 = None
        self._bm25_docs = []
        self._reranker = None

    def search(self, query: str, top_k: int = None,
               filter_expr: str = None, content_type: str = None) -> List[Dict[str, Any]]:
        """
        检索相关文档
        Args:
            query: 查询文本
            top_k: 返回数量
            filter_expr: Milvus 过滤表达式
            content_type: 按内容类型过滤 (text/table/image)
        Returns:
            检索结果列表，按相似度排序
        """
        top_k = top_k or self.top_k

        # 构造过滤条件
        if content_type and not filter_expr:
            filter_expr = f'content_type == "{content_type}"'
        elif content_type and filter_expr:
            filter_expr = f'{filter_expr} and content_type == "{content_type}"'

        results = []

        # 1. 向量检索
        vector_results = self._vector_search(query, top_k=top_k * 2, filter_expr=filter_expr)
        results.extend(vector_results)

        # 2. BM25 关键词检索（可选）
        if self.bm25_enabled:
            # 简单实现：如果数据量小，BM25 可以在内存中维护
            # 这里先只返回向量结果，后续完善
            pass

        # 3. 去重 & 排序
        results = self._merge_and_rank(results)

        # 4. Rerank 精排（可选）
        if self.rerank_enabled and len(results) > 1:
            results = self._rerank(query, results)

        # 5. 阈值过滤
        if self.similarity_threshold > 0:
            results = [r for r in results if r.get('score', 0) >= self.similarity_threshold]

        return results[:top_k]

    def _vector_search(self, query: str, top_k: int, filter_expr: str = None) -> List[Dict]:
        """向量检索"""
        query_vector = self.embedder.encode_query(query)

        results = self.vector_store.search(
            [query_vector],
            limit=top_k,
            filter_expr=filter_expr,
        )

        # 格式化结果
        formatted = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get('entity', {})
                formatted.append({
                    'score': hit.get('distance', 0),
                    'chunk_id': entity.get('chunk_id', ''),
                    'file_name': entity.get('file_name', ''),
                    'content': entity.get('content', ''),
                    'chapter_path': entity.get('chapter_path', ''),
                    'page_num': entity.get('page_num', 0),
                    'slide_num': entity.get('slide_num', 0),
                    'content_type': entity.get('content_type', 'text'),
                    'file_type': entity.get('file_type', ''),
                    'source': 'vector',
                })

        return formatted

    def _merge_and_rank(self, results: List[Dict]) -> List[Dict]:
        """
        合并多路检索结果并排序
        目前只有向量检索，直接按 score 排序即可
        后续加 BM25 后用 RRF 融合
        """
        # 按 chunk_id 去重，保留最高分
        seen = {}
        for r in results:
            cid = r.get('chunk_id')
            if cid not in seen or r.get('score', 0) > seen[cid].get('score', 0):
                seen[cid] = r

        # 按分数排序
        ranked = sorted(seen.values(), key=lambda x: x.get('score', 0), reverse=True)
        return ranked

    def _rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Reranker 精排
        使用 bge-reranker 对结果重新排序
        """
        if not self._reranker:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(retriever_config.RERANK_MODEL)
            except Exception as e:
                print(f"⚠️  Reranker 加载失败: {e}")
                return results

        # 构造 (query, doc) 对
        pairs = [[query, r.get('content', '')] for r in results]

        # 预测相似度
        scores = self._reranker.predict(pairs)

        # 更新分数并重排序
        for i, r in enumerate(results):
            r['rerank_score'] = float(scores[i])
            r['score'] = float(scores[i])  # 用 rerank 分数替代
            r['source'] = f"{r.get('source', '')}+rerank"

        ranked = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        return ranked


# 全局单例
_retriever = None


def get_retriever() -> Retriever:
    """获取检索器单例"""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
