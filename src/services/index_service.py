"""
索引服务 v2
- 文档入库 + 增量更新
- 支持：legacy parser / UnifiedParser(MarkItDown) / StructuredChunker v2
- 全链路 tracing（document_ingest / document_parse / chunk_create / embedding_store）
"""
import sys
import os
import time
import hashlib
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.embedding.embedder import get_embedding_model
from src.storage.vector_store import get_vector_store
from src.tracing.tracer import get_tracer
from config import parser_config


class IndexService:
    """索引服务"""

    def __init__(self, use_unified_parser: bool = None, use_structured_chunking: bool = None):
        self.embedder = get_embedding_model()
        self.vector_store = get_vector_store()
        self.tracer = get_tracer()
        self.use_unified_parser = (
            parser_config.USE_MARKITDOWN if use_unified_parser is None else use_unified_parser
        )
        self.use_structured_chunking = (
            parser_config.USE_STRUCTURED_CHUNKING if use_structured_chunking is None else use_structured_chunking
        )

    # ------------------------------------------------------------------
    def index_file(self, file_path: str) -> Dict:
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            return {'success': False, 'file_path': file_path, 'error': '文件不存在'}

        from src.parser import is_supported, get_supported_formats
        if not is_supported(file_path):
            return {'success': False, 'file_path': file_path,
                    'error': f'不支持: {get_supported_formats()}'}

        with self.tracer.span("document_ingest", stage="ingest") as ev:
            self.tracer.update_span_data(ev, file_path=file_path, file_name=os.path.basename(file_path))
            try:
                file_hash = self._get_file_hash(file_path)
                self.tracer.update_span_data(ev, file_hash=file_hash)

                # 已存在跳过
                existing = self._count_existing(file_hash)
                if existing > 0:
                    self.tracer.event("document_parse", data={"skipped": True, "existing_chunks": existing})
                    return {
                        'success': True, 'file_path': file_path, 'file_hash': file_hash,
                        'chunk_count': existing, 'skipped': True, 'message': '文件未变化，跳过'
                    }

                # 1. 解析
                print(f"📄 解析文件: {os.path.basename(file_path)}")
                with self.tracer.span("document_parse", stage="parse") as parse_ev:
                    chunks = self._parse_and_chunk(file_path, file_hash)
                    parser_used = "structured" if self.use_structured_chunking else "legacy"
                    self.tracer.update_span_data(
                        parse_ev, chunk_count=len(chunks),
                        parser_used=parser_used,
                        use_markitdown=self.use_unified_parser,
                    )
                print(f"   生成 {len(chunks)} 个分片")

                if not chunks:
                    return {'success': False, 'file_path': file_path, 'error': '解析结果为空'}

                # 2. 删除旧 chunk
                self._delete_by_file(file_path)

                # 3. embedding
                print(f"🔢 生成向量 ({len(chunks)} 条)...")
                with self.tracer.span("chunk_create", stage="chunk") as c_ev:
                    texts = [c['content'] for c in chunks]
                    self.tracer.update_span_data(c_ev, num_chunks=len(chunks))

                embeddings = self.embedder.encode(texts, show_progress_bar=False)

                # 4. 入库
                insert_data = []
                for i, ch in enumerate(chunks):
                    rec = {
                        'chunk_id': ch.get('chunk_id', f"{file_hash}_{i}"),
                        'embedding': embeddings[i].tolist(),
                        'file_name': ch.get('file_name', os.path.basename(file_path)),
                        'file_path': file_path,
                        'file_type': ch.get('file_type', os.path.splitext(file_path)[1].lstrip('.')),
                        'file_hash': file_hash,
                        'content_type': ch.get('content_type', 'text'),
                        'page_num': ch.get('page_num', ch.get('page', 0)),
                        'slide_num': ch.get('slide_num', 0),
                        'chapter_path': ch.get('chapter_path', ch.get('section_title', '')),
                        'chunk_index': i,
                        'content': ch.get('content', ''),
                        'create_time': time.time(),
                        'update_time': time.time(),
                        # 新增字段
                        'main_title': ch.get('main_title', ''),
                        'section_title': ch.get('section_title', ''),
                        'tokens': ch.get('tokens', 0),
                    }
                    insert_data.append(rec)

                with self.tracer.span("embedding_store", stage="store") as s_ev:
                    self.vector_store.insert(insert_data)
                    self.tracer.update_span_data(s_ev, stored=len(insert_data))

                print(f"✅ 索引完成: {len(insert_data)} 条")
                return {
                    'success': True, 'file_path': file_path,
                    'file_name': os.path.basename(file_path), 'file_hash': file_hash,
                    'chunk_count': len(insert_data),
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {'success': False, 'file_path': file_path, 'error': str(e)}

    # ------------------------------------------------------------------
    def _parse_and_chunk(self, file_path: str, file_hash: str):
        """
        返回 list of dict(chunk_id/file_name/file_type/content/.../main_title/section_title/page/tokens)
        """
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower().lstrip('.')

        if self.use_structured_chunking:
            from src.parser import parse_file_v2
            from src.chunking import StructuredChunker
            parsed = parse_file_v2(file_path)
            chunker = StructuredChunker()
            v2_chunks = chunker.chunk_document(parsed)
            out = []
            for c in v2_chunks:
                out.append({
                    'chunk_id': c.chunk_id,
                    'file_name': file_name,
                    'file_type': ext,
                    'content': c.content,
                    'main_title': c.main_title,
                    'section_title': c.section_title,
                    'chapter_path': c.section_path,
                    'page_num': c.page,
                    'page': c.page,
                    'tokens': c.tokens,
                })
            return out

        # legacy: 复用原有逻辑
        from src.parser import parse_file
        legacy = parse_file(file_path, use_unified=self.use_unified_parser)
        out = []
        for ch in legacy:
            meta = getattr(ch, 'metadata', {}) or {}
            out.append({
                'chunk_id': ch.chunk_id,
                'file_name': ch.file_name,
                'file_type': ch.file_type,
                'content': ch.content,
                'content_type': getattr(ch, 'content_type', 'text'),
                'page_num': getattr(ch, 'page_num', 0),
                'slide_num': getattr(ch, 'slide_num', 0),
                'chapter_path': getattr(ch, 'chapter_path', ''),
                'main_title': meta.get('main_title', ''),
                'section_title': meta.get('section_heading', ''),
                'tokens': 0,
            })
        return out

    # ------------------------------------------------------------------
    def index_directory(self, dir_path: str, recursive: bool = True) -> List[Dict]:
        from src.parser import is_supported
        dir_path = os.path.abspath(dir_path)
        results = []
        if not os.path.isdir(dir_path):
            print(f"❌ 目录不存在: {dir_path}")
            return results
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
        print(f"📂 找到 {len(files)} 个文件")
        for i, fp in enumerate(files, 1):
            print(f"[{i}/{len(files)}] ", end='')
            results.append(self.index_file(fp))
            print()
        return results

    def delete_file(self, file_path: str) -> bool:
        return self._delete_by_file(file_path) > 0

    def _count_existing(self, file_hash: str) -> int:
        try:
            existing = self.vector_store.client.query(
                collection_name=self.vector_store.collection_name,
                filter=f'file_hash == "{file_hash}"',
                output_fields=['chunk_id'], limit=1,
            ) if hasattr(self.vector_store, 'client') else []
            return len(existing or [])
        except Exception:
            return 0

    def _delete_by_file(self, file_path: str) -> int:
        file_hash = self._get_file_hash(file_path)
        try:
            existing = []
            if hasattr(self.vector_store, 'client'):
                existing = self.vector_store.client.query(
                    collection_name=self.vector_store.collection_name,
                    filter=f'file_hash == "{file_hash}"',
                    output_fields=['chunk_id'], limit=10000,
                )
            count = len(existing) if existing else 0
            if count > 0 or hasattr(self.vector_store, 'delete_by_file_hash'):
                self.vector_store.delete_by_file_hash(file_hash)
            return count
        except Exception:
            return 0

    def get_stats(self) -> Dict:
        return self.vector_store.get_stats()

    def list_indexed_files(self) -> List[Dict]:
        return self.vector_store.list_files()

    @staticmethod
    def _get_file_hash(file_path: str) -> str:
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()


_index_service = None

def get_index_service() -> IndexService:
    global _index_service
    if _index_service is None:
        _index_service = IndexService()
    return _index_service
