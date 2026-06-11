"""
索引服务 - 文档入库、增量更新
"""
import sys
import os
import time
import hashlib
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.parser import parse_file, is_supported, get_supported_formats
from src.embedding.embedder import get_embedding_model
from src.storage.vector_store import get_vector_store


class IndexService:
    """索引服务"""

    def __init__(self):
        self.embedder = get_embedding_model()
        self.vector_store = get_vector_store()

    def index_file(self, file_path: str) -> Dict:
        """
        索引单个文件
        Args:
            file_path: 文件路径
        Returns:
            索引结果 {success, file_name, chunk_count, error}
        """
        file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            return {
                'success': False,
                'file_path': file_path,
                'error': '文件不存在',
            }

        if not is_supported(file_path):
            return {
                'success': False,
                'file_path': file_path,
                'error': f'不支持的文件格式，支持: {get_supported_formats()}',
            }

        try:
            # 1. 计算文件 hash
            file_hash = self._get_file_hash(file_path)

            # 2. 检查是否已索引（hash 相同则跳过）
            existing = self.vector_store.client.query(
                collection_name=self.vector_store.collection_name,
                filter=f'file_hash == "{file_hash}"',
                output_fields=['count(*)' if False else 'chunk_id'],
                limit=1,
            )
            if len(existing) > 0:
                # 存在相同 hash 的文件，返回成功
                return {
                    'success': True,
                    'file_path': file_path,
                    'file_hash': file_hash,
                    'chunk_count': len(existing),
                    'skipped': True,
                    'message': '文件未变化，跳过',
                }

            # 3. 解析文档
            print(f"📄 解析文件: {os.path.basename(file_path)}")
            chunks = parse_file(file_path)
            print(f"   生成 {len(chunks)} 个分片")

            if not chunks:
                return {
                    'success': False,
                    'file_path': file_path,
                    'error': '解析结果为空',
                }

            # 4. 删除旧的 chunk（如果有）
            old_count = self._delete_by_file(file_path)
            if old_count > 0:
                print(f"   删除旧索引 {old_count} 条")

            # 5. 生成向量
            texts = [chunk.content for chunk in chunks]
            print(f"🔢 生成向量 ({len(texts)} 条)...")
            embeddings = self.embedder.encode(texts, show_progress_bar=False)

            # 6. 构造入库数据
            insert_data = []
            for i, chunk in enumerate(chunks):
                record = {
                    'chunk_id': chunk.chunk_id,
                    'embedding': embeddings[i].tolist(),
                    'file_name': chunk.file_name,
                    'file_path': chunk.file_path,
                    'file_type': chunk.file_type,
                    'file_hash': file_hash,
                    'content_type': chunk.content_type,
                    'page_num': chunk.page_num,
                    'slide_num': chunk.slide_num,
                    'chapter_path': chunk.chapter_path,
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'create_time': chunk.create_time,
                    'update_time': time.time(),
                }
                insert_data.append(record)

            # 7. 批量入库
            self.vector_store.insert(insert_data)

            print(f"✅ 索引完成: {len(insert_data)} 条")

            return {
                'success': True,
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_hash': file_hash,
                'chunk_count': len(insert_data),
            }

        except Exception as e:
            print(f"❌ 索引失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'file_path': file_path,
                'error': str(e),
            }

    def index_directory(self, dir_path: str, recursive: bool = True) -> List[Dict]:
        """
        索引目录下的所有文件
        Args:
            dir_path: 目录路径
            recursive: 是否递归子目录
        Returns:
            索引结果列表
        """
        dir_path = os.path.abspath(dir_path)
        results = []

        if not os.path.isdir(dir_path):
            print(f"❌ 目录不存在: {dir_path}")
            return results

        # 收集所有支持的文件
        files = []
        if recursive:
            for root, _, filenames in os.walk(dir_path):
                for fn in filenames:
                    fp = os.path.join(root, fn)
                    if is_supported(fp) and not fn.startswith('._'):
                        files.append(fp)
        else:
            for fn in os.listdir(dir_path):
                fp = os.path.join(dir_path, fn)
                if os.path.isfile(fp) and is_supported(fp) and not fn.startswith('._'):
                    files.append(fp)

        print(f"📂 找到 {len(files)} 个可索引文件:")
        for f in files:
            print(f"   - {os.path.relpath(f, dir_path)}")
        print()

        # 逐个索引
        success_count = 0
        for i, fp in enumerate(files, 1):
            print(f"[{i}/{len(files)}] ", end='')
            result = self.index_file(fp)
            results.append(result)
            if result.get('success'):
                success_count += 1
            print()

        print(f"\n{'=' * 60}")
        print(f"📊 索引完成: {success_count}/{len(files)} 成功")
        if success_count < len(files):
            for r in results:
                if not r.get('success'):
                    print(f"   ❌ {os.path.basename(r.get('file_path', ''))}: {r.get('error', '')}")

        return results

    def delete_file(self, file_path: str) -> bool:
        """删除指定文件的索引"""
        count = self._delete_by_file(file_path)
        print(f"删除 {file_path} 的 {count} 条索引")
        return count > 0

    def _delete_by_file(self, file_path: str) -> int:
        """按文件路径删除索引，返回删除数量"""
        file_hash = self._get_file_hash(file_path)

        try:
            # 先查询有多少条
            existing = self.vector_store.client.query(
                collection_name=self.vector_store.collection_name,
                filter=f'file_hash == "{file_hash}"',
                output_fields=['chunk_id'],
                limit=10000,
            )
            count = len(existing)

            if count > 0:
                self.vector_store.delete_by_file_hash(file_hash)

            return count
        except Exception:
            return 0

    def get_stats(self) -> Dict:
        """获取索引统计"""
        stats = self.vector_store.get_stats()
        return stats

    def list_indexed_files(self) -> List[Dict]:
        """列出已索引的文件"""
        return self.vector_store.list_files()

    @staticmethod
    def _get_file_hash(file_path: str) -> str:
        """计算文件 MD5"""
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()


# 全局单例
_index_service = None


def get_index_service() -> IndexService:
    """获取索引服务单例"""
    global _index_service
    if _index_service is None:
        _index_service = IndexService()
    return _index_service
