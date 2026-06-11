"""
文档解析器基类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import hashlib
import time


@dataclass
class DocumentChunk:
    """文档分片数据结构"""
    # 基础信息
    file_path: str
    file_name: str
    file_type: str
    file_hash: str

    # 内容
    chunk_id: str = ""
    content: str = ""
    content_type: str = "text"  # text / table / image_caption

    # 位置信息
    page_num: int = 0
    slide_num: int = 0
    chapter_path: str = ""
    chunk_index: int = 0

    # 元数据
    metadata: dict = field(default_factory=dict)
    create_time: float = field(default_factory=time.time)
    update_time: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"{self.file_hash}_{self.page_num}_{self.chunk_index}"


class BaseParser(ABC):
    """文档解析器基类"""

    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def parse(self, file_path: str) -> List[DocumentChunk]:
        """
        解析文档，返回分片列表
        Args:
            file_path: 文件路径
        Returns:
            DocumentChunk 列表
        """
        pass

    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """计算文件 MD5 哈希"""
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def split_text(self, text: str) -> List[str]:
        """
        简单的文本切片（按字符长度 + 重叠）
        子类可重写此方法实现更复杂的切片策略
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break

            start = end - self.chunk_overlap

        return chunks

    def extract_chapter(self, text: str, position: int) -> str:
        """提取当前位置的章节路径（可选实现）"""
        return ""
