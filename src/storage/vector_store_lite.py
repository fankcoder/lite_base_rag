"""
轻量向量存储 - 基于 numpy + JSON（适配低内存环境）
几千条向量以内完全够用，速度快，内存占用小
"""
import sys
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import milvus_config  # 复用配置名，实际用文件存储


class VectorStore:
    """轻量向量存储（numpy + JSON）"""

    def __init__(self, db_path=None, collection_name=None):
        self.db_path = Path(db_path or milvus_config.DB_PATH)
        self.collection_name = collection_name or milvus_config.COLLECTION_NAME
        self.vector_dim = milvus_config.VECTOR_DIM

        # 数据文件路径
        self._data_dir = self.db_path.parent / self.collection_name
        self._vectors_file = self._data_dir / "vectors.npy"
        self._metadata_file = self._data_dir / "metadata.json"

        # 内存中的数据
        self._vectors: Optional[np.ndarray] = None  # shape (n, dim)
        self._metadata: List[Dict] = []
        self._chunk_ids: List[str] = []

        # 加载数据
        self._load()

    def _load(self):
        """加载数据到内存"""
        if self._vectors_file.exists() and self._metadata_file.exists():
            self._vectors = np.load(self._vectors_file)
            with open(self._metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._metadata = data.get('metadata', [])
                self._chunk_ids = data.get('chunk_ids', [])
            print(f"📂 已加载向量库: {len(self._metadata)} 条")
        else:
            self._vectors = np.empty((0, self.vector_dim), dtype=np.float32)
            self._metadata = []
            self._chunk_ids = []
            print("📂 向量库为空（新建）")

    def _save(self):
        """保存到磁盘"""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        np.save(self._vectors_file, self._vectors)
        with open(self._metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': self._metadata,
                'chunk_ids': self._chunk_ids,
            }, f, ensure_ascii=False, indent=2)

    def create_collection(self, collection_name=None, drop_if_exists=False):
        """创建 collection（兼容 Milvus 接口）"""
        name = collection_name or self.collection_name

        if drop_if_exists and self._data_dir.exists():
            import shutil
            shutil.rmtree(self._data_dir)
            print(f"已删除旧 Collection: {name}")

        if self._data_dir.exists():
            print(f"Collection {name} 已存在")
            return

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._vectors = np.empty((0, self.vector_dim), dtype=np.float32)
        self._metadata = []
        self._chunk_ids = []
        self._save()

        print(f"✅ Collection {name} 创建成功")
        print(f"   - 向量维度: {self.vector_dim}")
        print(f"   - 存储方式: numpy + JSON (轻量模式)")

    def insert(self, data: List[Dict], collection_name=None):
        """
        插入数据
        data: [{'chunk_id': ..., 'embedding': [...], 'file_name': ...}, ...]
        """
        if not data:
            return 0

        # 检查是否已存在相同 chunk_id
        new_vectors = []
        new_metadata = []
        new_chunk_ids = []

        existing_set = set(self._chunk_ids)

        for item in data:
            cid = item.get('chunk_id')
            if cid in existing_set:
                # 已存在则跳过（如需更新请先删除）
                continue

            emb = item.pop('embedding', None)
            if emb is None:
                continue

            new_vectors.append(emb)
            new_metadata.append(item)
            new_chunk_ids.append(cid)

        if not new_vectors:
            return 0

        # 转为 numpy 数组
        new_vecs_np = np.array(new_vectors, dtype=np.float32)

        # 追加
        if self._vectors is None or len(self._vectors) == 0:
            self._vectors = new_vecs_np
        else:
            self._vectors = np.vstack([self._vectors, new_vecs_np])

        self._metadata.extend(new_metadata)
        self._chunk_ids.extend(new_chunk_ids)

        # 保存
        self._save()

        return len(new_vectors)

    def search(self, query_vectors: List[List[float]], limit: int = 10,
               filter_expr: str = None, output_fields: List[str] = None,
               collection_name=None):
        """
        向量检索（余弦相似度）
        返回格式兼容 Milvus
        """
        if self._vectors is None or len(self._vectors) == 0:
            return [[]]

        results = []

        for query_vec in query_vectors:
            query_np = np.array(query_vec, dtype=np.float32)

            # 计算余弦相似度（向量已归一化时 = 点积）
            # similarity = dot(a, b) / (||a|| * ||b||)
            # 这里假设向量已归一化，直接点积
            similarities = np.dot(self._vectors, query_np)

            # 过滤（简单实现，支持 == 比较）
            mask = None
            if filter_expr:
                mask = self._parse_filter(filter_expr)

            if mask is not None and len(mask) > 0:
                # 过滤后的索引
                indices = np.array(mask)
                if len(indices) == 0:
                    results.append([])
                    continue
                sims = similarities[indices]
                metas = [self._metadata[i] for i in indices]
            else:
                indices = np.arange(len(similarities))
                sims = similarities
                metas = self._metadata

            # Top-K
            top_k = min(limit, len(sims))
            top_indices = np.argsort(sims)[::-1][:top_k]

            # 格式化结果（兼容 Milvus 格式）
            hits = []
            for idx in top_indices:
                meta = metas[idx]
                # 只返回请求的字段
                if output_fields:
                    entity = {k: meta.get(k, '') for k in output_fields if k in meta}
                else:
                    entity = dict(meta)

                hits.append({
                    'distance': float(sims[idx]),
                    'entity': entity,
                })

            results.append(hits)

        return results

    def _parse_filter(self, filter_expr: str) -> Optional[List[int]]:
        """
        简单的过滤表达式解析
        支持: field == "value"  和  and  连接
        """
        if not filter_expr:
            return None

        # 拆分 and
        conditions = [c.strip() for c in filter_expr.split(' and ')]
        matched_indices = list(range(len(self._metadata)))

        for cond in conditions:
            # 解析 field == "value"
            if ' == ' in cond:
                field, value = cond.split(' == ', 1)
                field = field.strip()
                value = value.strip().strip('"').strip("'")

                new_matched = []
                for idx in matched_indices:
                    meta = self._metadata[idx]
                    if str(meta.get(field, '')) == value:
                        new_matched.append(idx)
                matched_indices = new_matched

        return matched_indices if matched_indices else []

    def delete_by_file_hash(self, file_hash: str, collection_name=None):
        """按文件 hash 删除"""
        if not self._metadata:
            return 0

        keep_indices = []
        delete_count = 0

        for i, meta in enumerate(self._metadata):
            if meta.get('file_hash') == file_hash:
                delete_count += 1
            else:
                keep_indices.append(i)

        if delete_count > 0:
            self._vectors = self._vectors[keep_indices]
            self._metadata = [self._metadata[i] for i in keep_indices]
            self._chunk_ids = [self._chunk_ids[i] for i in keep_indices]
            self._save()

        return delete_count

    def get_stats(self, collection_name=None):
        """获取统计信息"""
        return {
            'row_count': len(self._metadata) if self._metadata else 0,
            'vector_dim': self.vector_dim,
            'storage_type': 'numpy+json (lite)',
        }

    def list_files(self, collection_name=None):
        """列出已索引的文件（去重）"""
        seen = {}
        for meta in self._metadata:
            fh = meta.get('file_hash')
            if fh and fh not in seen:
                seen[fh] = {
                    'file_name': meta.get('file_name', ''),
                    'file_hash': fh,
                    'file_type': meta.get('file_type', ''),
                    'create_time': meta.get('create_time', 0),
                }
        return list(seen.values())

    def has_collection(self, collection_name=None):
        """检查是否存在"""
        return self._data_dir.exists()

    def drop_collection(self, collection_name=None):
        """删除 collection"""
        import shutil
        if self._data_dir.exists():
            shutil.rmtree(self._data_dir)
            self._vectors = np.empty((0, self.vector_dim), dtype=np.float32)
            self._metadata = []
            self._chunk_ids = []
            print(f"Collection {collection_name or self.collection_name} 已删除")

    def close(self):
        """保存并关闭"""
        self._save()


# 全局单例
_vector_store = None


def get_vector_store() -> VectorStore:
    """获取向量存储单例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
