"""
文档解析器工厂
"""
import os
from typing import Dict, Type

from .base_parser import BaseParser, DocumentChunk
from .md_parser import MDParser
from .pdf_parser import PDFParser
from .ppt_parser import PPTParser


# 格式 -> 解析器映射
_PARSER_REGISTRY: Dict[str, Type[BaseParser]] = {
    '.md': MDParser,
    '.markdown': MDParser,
    '.pdf': PDFParser,
    '.pptx': PPTParser,
}


def get_parser(file_path: str, **kwargs) -> BaseParser:
    """
    根据文件扩展名获取对应的解析器
    Args:
        file_path: 文件路径
        **kwargs: 传递给解析器的参数
    Returns:
        BaseParser 实例
    Raises:
        ValueError: 不支持的文件格式
    """
    ext = os.path.splitext(file_path)[1].lower()

    parser_class = _PARSER_REGISTRY.get(ext)
    if parser_class is None:
        raise ValueError(f"不支持的文件格式: {ext}")

    return parser_class(**kwargs)


def parse_file(file_path: str, **kwargs):
    """
    便捷函数：解析单个文件
    Args:
        file_path: 文件路径
        **kwargs: 解析器参数
    Returns:
        DocumentChunk 列表
    """
    parser = get_parser(file_path, **kwargs)
    return parser.parse(file_path)


def is_supported(file_path: str) -> bool:
    """检查文件是否支持解析"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in _PARSER_REGISTRY


def get_supported_formats() -> list:
    """获取支持的格式列表"""
    return list(_PARSER_REGISTRY.keys())


__all__ = [
    'BaseParser',
    'DocumentChunk',
    'MDParser',
    'PDFParser',
    'PPTParser',
    'get_parser',
    'parse_file',
    'is_supported',
    'get_supported_formats',
]
