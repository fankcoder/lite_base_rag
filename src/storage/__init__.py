"""
存储模块
默认使用 Milvus Lite（零配置、单文件）
"""
from .vector_store import VectorStore, get_vector_store

# 如果要用轻量 numpy 版本（低内存环境），取消下面注释：
# from .vector_store_lite import VectorStore, get_vector_store

__all__ = ["VectorStore", "get_vector_store"]
