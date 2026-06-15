"""
Chunking v2 - 结构化语义切片
- 优先使用 ParsedDocument 的 section 结构（不跨 section）
- 每个 chunk 自动注入 main_title + section_title（防语义漂移）
- 支持 code block / list 识别，尽量不破坏其完整性
- fallback：无结构时使用 sliding window
"""
from .chunker import (
    StructuredChunker,
    ChunkV2,
    ChunkingConfig,
)

__all__ = ["StructuredChunker", "ChunkV2", "ChunkingConfig"]
