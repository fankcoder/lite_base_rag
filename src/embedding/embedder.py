"""
Embedding 模块 - 文本向量化
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import embedding_config


class EmbeddingModel:
    """文本 Embedding 模型封装"""

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or embedding_config.MODEL_NAME
        self.device = device or embedding_config.DEVICE
        self.normalize = embedding_config.NORMALIZE_EMBEDDINGS
        self.max_seq_length = embedding_config.MAX_SEQ_LENGTH

        self._model = None
        self._dimension = None

    def _load_model(self):
        """懒加载模型"""
        if self._model is None:
            print(f"📥 加载 Embedding 模型: {self.model_name} ({self.device})")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            self._model.max_seq_length = self.max_seq_length
            self._dimension = self._model.get_sentence_embedding_dimension()
            print(f"   维度: {self._dimension}")

    @property
    def model(self):
        self._load_model()
        return self._model

    @property
    def dimension(self):
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def encode(self, texts, batch_size=None, show_progress_bar=False):
        """
        文本向量化
        Args:
            texts: 文本或文本列表
            batch_size: 批大小
            show_progress_bar: 显示进度条
        Returns:
            numpy array, shape (n_texts, dim)
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or embedding_config.BATCH_SIZE

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

        return embeddings

    def encode_query(self, query):
        """
        编码查询（BGE 系列需要加指令前缀）
        """
        # BGE 模型的 query 指令前缀
        if "bge" in self.model_name.lower():
            query = f"为这个句子生成表示以用于检索相关文章：{query}"

        return self.encode(query)[0]

    def similarity(self, text1, text2):
        """计算两个文本的余弦相似度"""
        emb1, emb2 = self.encode([text1, text2])
        import numpy as np
        return float(np.dot(emb1, emb2))


# 全局单例
_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """获取 Embedding 模型单例"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
