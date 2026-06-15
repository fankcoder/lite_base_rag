"""
Unified Parser Adapter
- 优先使用 Microsoft MarkItDown（markitdown）把 PDF/DOCX/PPTX/HTML/MD 转成 Markdown
- 转换失败时 fallback 到现有遗留解析器
- 再把 markdown 解析为统一结构 ParsedDocument{title, sections[]}
  sections[i] = {heading, content, page}
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_parser import BaseParser, DocumentChunk

logger = logging.getLogger("rag.parser.unified")


# ---------- 统一输出结构 ----------

@dataclass
class ParsedSection:
    heading: str = ""
    content: str = ""
    page: int = 0
    level: int = 0  # 标题层级 H1=1 H2=2 ...


@dataclass
class ParsedDocument:
    title: str = ""
    file_path: str = ""
    file_name: str = ""
    file_type: str = ""
    file_hash: str = ""
    sections: List[ParsedSection] = field(default_factory=list)
    raw_markdown: str = ""
    parser_used: str = ""  # "markitdown" / "legacy"


class MarkItDownAdapter:
    """Microsoft MarkItDown 的封装，带 lazy import，未安装时优雅降级"""

    def __init__(self):
        self._md = None
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                from markitdown import MarkItDown  # noqa: F401
                self._available = True
            except Exception as e:
                logger.warning("markitdown not available: %s (fallback to legacy parsers)", e)
                self._available = False
        return self._available

    def _get_instance(self):
        if self._md is None and self.available:
            from markitdown import MarkItDown
            self._md = MarkItDown()
        return self._md

    def parse(self, file_path: str) -> Optional[str]:
        """
        返回 markdown 文本；失败返回 None
        """
        if not self.available:
            return None
        try:
            md = self._get_instance()
            result = md.convert(file_path)
            text = getattr(result, "text_content", None) or (result.get("text_content") if isinstance(result, dict) else None)
            if not text or not text.strip():
                return None
            return text
        except Exception as e:
            logger.warning("markitdown parse failed for %s: %s", file_path, e)
            return None


# ---------- Markdown -> ParsedDocument ----------

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*$', re.MULTILINE)
_PAGE_HINT_RE = re.compile(r'(?:<!--\s*page\s*:\s*(\d+)\s*-->|<!--\s*slide\s*:\s*(\d+)\s*-->|\[p\.?\s*(\d+)\])', re.IGNORECASE)


def markdown_to_sections(md_text: str) -> ParsedDocument:
    """
    将 markdown 按标题切分成 sections
    支持 # H1 / ## H2 / ### H3 ...
    """
    doc = ParsedDocument(raw_markdown=md_text)
    if not md_text:
        return doc

    # 找所有 heading 位置
    headings = list(_HEADING_RE.finditer(md_text))

    if not headings:
        # 无标题文档，整体作为一个 section，title 取第一行
        first_line = md_text.strip().splitlines()[0][:120] if md_text.strip() else ""
        doc.title = first_line
        doc.sections.append(ParsedSection(heading=first_line, content=md_text.strip(), page=0))
        return doc

    # document title = 第一个 H1，或第一个 heading
    first = headings[0]
    doc.title = first.group(2).strip()

    # 切分 section
    for i, h in enumerate(headings):
        start = h.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(md_text)
        body = md_text[start:end].strip()
        level = len(h.group(1))
        heading = h.group(2).strip()

        if not body and i + 1 < len(headings) and len(headings[i + 1].group(1)) > level:
            # 父级标题（仅容器，无正文），跳过作为独立 section
            continue
        if not body:
            continue

        # 尝试从 body 提取 page hint
        page = 0
        m = _PAGE_HINT_RE.search(body)
        if m:
            for g in m.groups():
                if g:
                    page = int(g)
                    break

        doc.sections.append(
            ParsedSection(heading=heading, content=body, page=page, level=level)
        )

    return doc


# ---------- 统一 Parser ----------

class UnifiedParser(BaseParser):
    """
    对外接口：
    - parse(path) -> List[DocumentChunk] (兼容旧接口，使用简单切片，避免破坏存量调用方)
    - parse_document(path) -> ParsedDocument  (结构化输出，供 StructuredChunker 使用)
    """

    SUPPORTED_EXTS = {".md", ".markdown", ".pdf", ".pptx", ".docx", ".html", ".htm", ".txt"}

    def __init__(self, chunk_size=500, chunk_overlap=100, enable_markitdown: bool = True, **_):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._markitdown = MarkItDownAdapter() if enable_markitdown else None
        self.enable_markitdown = enable_markitdown

    # ---- 旧接口兼容：返回 DocumentChunk 列表 ----
    def parse(self, file_path: str) -> List[DocumentChunk]:
        doc = self.parse_document(file_path)
        file_hash = self.get_file_hash(file_path)
        chunks: List[DocumentChunk] = []
        idx = 0
        for sec in doc.sections:
            for piece in self.split_text(sec.content):
                chunks.append(DocumentChunk(
                    file_path=file_path,
                    file_name=os.path.basename(file_path),
                    file_type=os.path.splitext(file_path)[1].lower().lstrip("."),
                    file_hash=file_hash,
                    chunk_id=f"{file_hash}_{sec.page}_{idx}",
                    content=piece,
                    page_num=sec.page,
                    chapter_path=sec.heading,
                    chunk_index=idx,
                    metadata={"section_heading": sec.heading, "main_title": doc.title, "parser": doc.parser_used},
                ))
                idx += 1
        return chunks

    # ---- 新接口：返回结构化 ParsedDocument ----
    def parse_document(self, file_path: str) -> ParsedDocument:
        ext = os.path.splitext(file_path)[1].lower()
        file_hash = self.get_file_hash(file_path)

        md_text: Optional[str] = None
        parser_used = "legacy"

        # 1. 优先 markitdown
        if self.enable_markitdown and self._markitdown and self._markitdown.available:
            md_text = self._markitdown.parse(file_path)
            if md_text is not None:
                parser_used = "markitdown"

        # 2. Fallback：各 legacy parser 结果 -> markdown
        if md_text is None:
            try:
                legacy_chunks = self._legacy_parse(file_path)
                # 拼接 legacy 结果成 markdown，附加标题
                md_text = self._chunks_to_markdown(legacy_chunks)
                parser_used = "legacy"
            except Exception as e:
                logger.warning("legacy parser failed: %s", e)
                md_text = ""

        doc = markdown_to_sections(md_text or "")
        doc.file_path = file_path
        doc.file_name = os.path.basename(file_path)
        doc.file_type = ext.lstrip(".")
        doc.file_hash = file_hash
        doc.parser_used = parser_used

        if not doc.title:
            doc.title = os.path.splitext(doc.file_name)[0]

        return doc

    # ---- 内部：legacy parser 调度 ----
    def _legacy_parse(self, file_path: str) -> List[DocumentChunk]:
        from .md_parser import MDParser
        from .pdf_parser import PDFParser
        from .ppt_parser import PPTParser

        ext = os.path.splitext(file_path)[1].lower()
        mapping: Dict[str, Any] = {
            ".md": MDParser, ".markdown": MDParser,
            ".pdf": PDFParser,
            ".pptx": PPTParser,
        }
        cls = mapping.get(ext)
        if cls is None:
            # 纯文本
            text_chunks: List[DocumentChunk] = []
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for i, piece in enumerate(self.split_text(content)):
                text_chunks.append(DocumentChunk(
                    file_path=file_path,
                    file_name=os.path.basename(file_path),
                    file_type=ext.lstrip("."),
                    file_hash=self.get_file_hash(file_path),
                    chunk_id=f"text_{i}",
                    content=piece,
                    chunk_index=i,
                ))
            return text_chunks
        inst = cls(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return inst.parse(file_path)

    def _chunks_to_markdown(self, chunks: List[DocumentChunk]) -> str:
        if not chunks:
            return ""
        lines: List[str] = []
        for c in chunks:
            title = c.chapter_path or f"第 {c.page_num or c.slide_num or 0} 页/节"
            lines.append(f"## {title}")
            lines.append("")
            lines.append(c.content)
            lines.append("")
        return "\n".join(lines)
