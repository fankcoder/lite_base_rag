"""
Parser 层重新导出
新增 MarkItDown 统一 adapter 作为首选解析器，原有 parser 保留为 fallback
"""
from .base_parser import BaseParser, DocumentChunk
from .md_parser import MDParser
from .pdf_parser import PDFParser
from .ppt_parser import PPTParser

# 新的统一 parser adapter（markitdown + fallback）
from .unified_parser import UnifiedParser, ParsedSection, ParsedDocument

# 新 chunking v2
from ..chunking import StructuredChunker, ChunkV2

import os
from typing import List


_PARSER_REGISTRY = {
    '.md': MDParser,
    '.markdown': MDParser,
    '.pdf': PDFParser,
    '.pptx': PPTParser,
}


def get_parser(file_path: str, use_unified: bool = True, **kwargs):
    ext = os.path.splitext(file_path)[1].lower()
    if use_unified:
        # UnifiedParser 内部按 ext 调度；markitdown 失败自动 fallback
        return UnifiedParser(**kwargs)
    cls = _PARSER_REGISTRY.get(ext)
    if cls is None:
        raise ValueError(f"不支持的文件格式: {ext}")
    return cls(**kwargs)


def parse_file(file_path: str, use_unified: bool = True, **kwargs) -> List[DocumentChunk]:
    p = get_parser(file_path, use_unified=use_unified, **kwargs)
    return p.parse(file_path)


def parse_file_v2(file_path: str, **kwargs):
    """返回 ParsedDocument（结构化），供 StructuredChunker 使用"""
    p = UnifiedParser(**kwargs)
    return p.parse_document(file_path)


def is_supported(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    return ext in _PARSER_REGISTRY or ext in {'.docx', '.html', '.htm', '.txt'}


def get_supported_formats() -> list:
    return list({*_PARSER_REGISTRY.keys(), '.docx', '.html', '.htm', '.txt'})


__all__ = [
    'BaseParser', 'DocumentChunk',
    'MDParser', 'PDFParser', 'PPTParser',
    'UnifiedParser', 'ParsedDocument', 'ParsedSection',
    'StructuredChunker', 'ChunkV2',
    'get_parser', 'parse_file', 'parse_file_v2', 'is_supported', 'get_supported_formats',
]
