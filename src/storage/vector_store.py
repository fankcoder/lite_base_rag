"""
存储层 - Milvus 向量数据库封装
"""
import sys
from pathlib import Path

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import milvus_config


class VectorStore:
    """Milvus Lite 向量存储封装"""

    def __init__(self, db_path=None, collection_name=None):
        self.db_path = db_path or milvus_config.DB_PATH
        self.collection_name = collection_name or milvus_config.COLLECTION_NAME
        self.vector_dim = milvus_config.VECTOR_DIM

        # 确保目录存在
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # 连接 Milvus Lite
        self.client = MilvusClient(uri=self.db_path)
        
        # 如果 collection 存在，自动加载
        if self.has_collection():
            try:
                self.client.load_collection(self.collection_name)
            except Exception:
                pass

    def create_collection(self, collection_name=None, drop_if_exists=False):
        """创建 Collection"""
        name = collection_name or self.collection_name

        if drop_if_exists and self.client.has_collection(name):
            self.client.drop_collection(name)
            print(f"已删除旧 Collection: {name}")

        if self.client.has_collection(name):
            print(f"Collection {name} 已存在")
            try:
                self.client.load_collection(name)
            except Exception:
                pass
            return

        # Milvus Lite 简单模式：自动创建 schema 和索引
        # 主键用 chunk_id (VARCHAR)，向量维度 512
        self.client.create_collection(
            collection_name=name,
            dimension=self.vector_dim,
            primary_field_name="chunk_id",
            id_type="string",
            vector_field_name="embedding",
            metric_type=milvus_config.METRIC_TYPE,
            auto_id=False,
            enable_dynamic_field=True,  # 动态字段，元数据随便加
        )

        # 加载 collection
        self.client.load_collection(name)

        print(f"✅ Collection {name} 创建成功")
        print(f"   - 向量维度: {self.vector_dim}")
        print(f"   - 距离度量: {milvus_config.METRIC_TYPE}")
        print(f"   - 动态字段: 启用（元数据灵活扩展）")

    def insert(self, data, collection_name=None):
        """插入数据"""
        name = collection_name or self.collection_name
        result = self.client.insert(collection_name=name, data=data)
        return result

    def search(self, query_vectors, limit=10, filter_expr=None,
               output_fields=None, collection_name=None):
        """向量检索"""
        name = collection_name or self.collection_name
        output_fields = output_fields or [
            "chunk_id", "file_name", "content", "chapter_path",
            "page_num", "slide_num", "content_type", "file_type"
        ]

        results = self.client.search(
            collection_name=name,
            data=query_vectors,
            limit=limit,
            filter=filter_expr,
            output_fields=output_fields,
            search_params={
                "metric_type": milvus_config.METRIC_TYPE,
                "params": {"nprobe": 16},
            }
        )

        return results

    def delete_by_file_hash(self, file_hash, collection_name=None):
        """按文件 hash 删除所有 chunk"""
        name = collection_name or self.collection_name
        result = self.client.delete(
            collection_name=name,
            filter=f'file_hash == "{file_hash}"',
        )
        return result

    def get_stats(self, collection_name=None):
        """获取统计信息"""
        name = collection_name or self.collection_name
        stats = self.client.get_collection_stats(name)
        return stats

    def list_files(self, collection_name=None):
        """获取已索引的文件列表（去重）"""
        name = collection_name or self.collection_name
        # Milvus Lite 不支持聚合查询，这里用简单方式
        # 实际使用时可以配合 SQLite 存储元数据
        results = self.client.query(
            collection_name=name,
            filter="",
            output_fields=["file_name", "file_hash", "file_type", "create_time"],
            limit=10000,
        )
        # 按 file_hash 去重
        seen = {}
        for item in results:
            fh = item.get("file_hash")
            if fh and fh not in seen:
                seen[fh] = item
        return list(seen.values())

    def has_collection(self, collection_name=None):
        """检查 collection 是否存在"""
        name = collection_name or self.collection_name
        return self.client.has_collection(name)

    def drop_collection(self, collection_name=None):
        """删除 collection"""
        name = collection_name or self.collection_name
        if self.has_collection(name):
            self.client.drop_collection(name)
            print(f"Collection {name} 已删除")

    def close(self):
        """关闭连接"""
        # Milvus Lite 不需要显式关闭
        pass


# 全局单例
_vector_store = None


def get_vector_store() -> VectorStore:
    """获取向量存储单例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
