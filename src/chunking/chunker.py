"""
结构化 Chunker v2
核心约束：
- 不跨 section
- 每个 chunk 带 main_title + section_title（防语义漂移）
- code block / 列表保持完整
- 无结构时 fallback sliding window
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from ..parser.unified_parser import ParsedDocument, ParsedSection
except Exception:  # pragma: no cover - fallback for direct script execution
    pass


@dataclass
class ChunkingConfig:
    max_tokens: int = 500          # 单 chunk 最大 token 数（粗估）
    min_tokens: int = 50           # 小于此值时与相邻 chunk 合并
    overlap_tokens: int = 80       # sliding window 重叠
    prefer_section_boundary: bool = True
    preserve_blocks: bool = True   # 保留 code / list 完整性
    chars_per_token: float = 1.6   # 中文字符粗略换算


@dataclass
class ChunkV2:
    doc_id: str
    chunk_id: str
    main_title: str
    section_title: str
    content: str
    position: str              # "main_title#section_title#idx"
    page: int = 0
    tokens: int = 0
    section_path: str = ""     # H1 > H2 > H3 ...
    metadata: dict = field(default_factory=dict)
    create_time: float = field(default_factory=time.time)


# ---------- 工具：粗略 token 估算 ----------

def _estimate_tokens(text: str, chars_per_token: float = 1.6) -> int:
    if not text:
        return 0
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    ascii_words = len(re.findall(r'[A-Za-z0-9_]+', text))
    return int((chinese + ascii_words) / max(0.5, chars_per_token - 0.6) + ascii_words * 0.4)


_CODE_FENCE_RE = re.compile(r'^```', re.MULTILINE)
_LIST_LINE_RE = re.compile(r'^\s*(?:[-*+]|\d+\.)\s+')


class StructuredChunker:
    """结构化 chunker"""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    # ---------- 公开接口 ----------

    def chunk_document(self, doc: "ParsedDocument") -> List[ChunkV2]:
        """
        将 ParsedDocument 切分为 ChunkV2 列表
        """
        cfg = self.config
        doc_id = doc.file_hash or hashlib.md5(doc.file_name.encode()).hexdigest()[:16]
        chunks: List[ChunkV2] = []

        if not doc.sections:
            # fallback：整个文档作为一个 section，走 sliding window
            return self._sliding_window(
                doc_id=doc_id,
                main_title=doc.title or doc.file_name,
                section_title="",
                content=doc.raw_markdown or "",
                page=0,
                section_path=doc.title or doc.file_name,
            )

        for sec_idx, sec in enumerate(doc.sections):
            content = sec.content.strip()
            section_title = sec.heading.strip()
            section_path = f"{doc.title} > {section_title}" if doc.title else section_title

            if not content:
                continue

            estimated = _estimate_tokens(content, cfg.chars_per_token)

            # 单 section 足够小：一个 chunk
            if estimated <= cfg.max_tokens and estimated >= cfg.min_tokens:
                chunks.append(self._make_chunk(
                    doc_id=doc_id, main_title=doc.title or doc.file_name,
                    section_title=section_title, content=content,
                    page=sec.page, idx=len(chunks), section_path=section_path,
                ))
                continue

            # section 过短：尝试与相邻 section 合并（前提是相邻 section 标题同一父级）
            if estimated < cfg.min_tokens:
                merged = self._try_merge_small(doc, sec_idx, chunks)
                if merged is not None:
                    chunks.append(merged)
                    continue

            # 大 section：按块切，保留 code/list 完整性
            sub_chunks = self._split_large_section(content, cfg)
            for i, piece in enumerate(sub_chunks):
                sub_title = section_title if i == 0 else f"{section_title} (续{i})"
                chunks.append(self._make_chunk(
                    doc_id=doc_id, main_title=doc.title or doc.file_name,
                    section_title=sub_title, content=piece,
                    page=sec.page, idx=len(chunks), section_path=section_path,
                ))

        # 二次扫描：最后 <min_tokens 的 chunk 并入前一个
        chunks = self._merge_tails(chunks, cfg)
        # 重新生成 chunk_id（基于序号保证稳定）
        for i, c in enumerate(chunks):
            c.chunk_id = f"{doc_id}_{i:05d}"
            c.position = f"{c.main_title}#{c.section_title}#{i}"
        return chunks

    # ---------- 内部 ----------

    def _make_chunk(self, doc_id: str, main_title: str, section_title: str,
                    content: str, page: int, idx: int, section_path: str) -> ChunkV2:
        # 注入标题上下文：每个 chunk 自带 main_title + section_title，防语义漂移
        contextual = f"# {main_title}\n## {section_title}\n\n{content}" if section_title else f"# {main_title}\n\n{content}"
        tokens = _estimate_tokens(contextual, self.config.chars_per_token)
        return ChunkV2(
            doc_id=doc_id,
            chunk_id=f"{doc_id}_{idx:05d}",
            main_title=main_title,
            section_title=section_title,
            content=contextual,
            position=f"{main_title}#{section_title}#{idx}",
            page=page,
            tokens=tokens,
            section_path=section_path,
            metadata={"page": page, "raw_content": content},
        )

    def _try_merge_small(self, doc, sec_idx: int, existing: List[ChunkV2]) -> Optional[ChunkV2]:
        """小节合并到上一个 chunk（若同 page / 同 level）"""
        sec = doc.sections[sec_idx]
        if not existing:
            # 第一个 section 太小，仍然生成一个 chunk，后续可能被 merge tail 处理
            return self._make_chunk(
                doc_id=doc.file_hash, main_title=doc.title or doc.file_name,
                section_title=sec.heading, content=sec.content.strip(),
                page=sec.page, idx=len(existing),
                section_path=f"{doc.title} > {sec.heading}",
            )
        prev = existing[-1]
        if prev.page != sec.page or prev.section_path.split(">")[-1].strip() == sec.heading:
            return None
        combined_raw = (prev.metadata.get("raw_content", "") + "\n\n" + sec.content.strip()).strip()
        if _estimate_tokens(combined_raw) > self.config.max_tokens:
            return None
        # 直接替换前一个
        prev.content = f"# {prev.main_title}\n## {prev.section_title}\n\n{combined_raw}"
        prev.metadata["raw_content"] = combined_raw
        prev.tokens = _estimate_tokens(prev.content, self.config.chars_per_token)
        return None

    def _merge_tails(self, chunks: List[ChunkV2], cfg: ChunkingConfig) -> List[ChunkV2]:
        out: List[ChunkV2] = []
        for c in chunks:
            if out and c.tokens < cfg.min_tokens:
                prev = out[-1]
                prev_raw = prev.metadata.get("raw_content", "")
                cur_raw = c.metadata.get("raw_content", "")
                new_raw = (prev_raw + "\n\n" + cur_raw).strip()
                if _estimate_tokens(new_raw) <= cfg.max_tokens:
                    prev.metadata["raw_content"] = new_raw
                    prev.content = f"# {prev.main_title}\n## {prev.section_title}\n\n{new_raw}"
                    prev.tokens = _estimate_tokens(prev.content)
                    continue
            out.append(c)
        return out

    def _split_large_section(self, content: str, cfg: ChunkingConfig) -> List[str]:
        """
        切分大 section：按段落、code block、列表边界切；超过 max_tokens 的段落再 sliding window
        """
        blocks = self._split_into_blocks(content) if cfg.preserve_blocks else [content]
        pieces: List[str] = []
        buf: List[str] = []
        buf_tokens = 0

        for blk in blocks:
            blk_tokens = _estimate_tokens(blk, cfg.chars_per_token)
            # 单块本身超过 max_tokens -> sliding window 子切
            if blk_tokens > cfg.max_tokens:
                if buf:
                    pieces.append("\n\n".join(buf).strip())
                    buf, buf_tokens = [], 0
                for sub in self._sliding_text(blk, cfg.max_tokens, cfg.overlap_tokens):
                    pieces.append(sub)
                continue
            if buf_tokens + blk_tokens > cfg.max_tokens and buf:
                pieces.append("\n\n".join(buf).strip())
                buf, buf_tokens = [blk], blk_tokens
            else:
                buf.append(blk)
                buf_tokens += blk_tokens
        if buf:
            pieces.append("\n\n".join(buf).strip())
        return [p for p in pieces if p.strip()]

    def _split_into_blocks(self, content: str) -> List[str]:
        """按 code fences / 空行 / 列表起始做块切分"""
        lines = content.splitlines()
        blocks: List[str] = []
        buf: List[str] = []
        in_code = False

        def flush():
            if buf:
                blocks.append("\n".join(buf).strip())
                buf.clear()

        for line in lines:
            if _CODE_FENCE_RE.match(line):
                in_code = not in_code
                buf.append(line)
                continue
            if in_code:
                buf.append(line)
                continue
            if line.strip() == "":
                flush()
                continue
            if _LIST_LINE_RE.match(line) and buf and not _LIST_LINE_RE.match(buf[-1]):
                flush()
                buf.append(line)
            else:
                buf.append(line)
        flush()
        return blocks

    def _sliding_text(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        """纯字符级 sliding window（fallback），按中文 1.6 char/token 估算"""
        chars_per_token = self.config.chars_per_token
        max_chars = int(max_tokens * chars_per_token)
        overlap_chars = int(overlap * chars_per_token)
        if len(text) <= max_chars:
            return [text.strip()]
        pieces = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            # 尽量在换行/句号处断开
            cut = text.rfind("\n", start, end)
            if cut == -1 or cut < start + max_chars // 2:
                cut = text.rfind("。", start, end)
            if cut != -1 and cut > start + max_chars // 2:
                end = cut + 1
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= len(text):
                break
            start = max(end - overlap_chars, start + 1)
        return pieces

    def _sliding_window(self, doc_id, main_title, section_title, content, page, section_path) -> List[ChunkV2]:
        cfg = self.config
        pieces = self._sliding_text(content, cfg.max_tokens, cfg.overlap_tokens)
        out: List[ChunkV2] = []
        for i, p in enumerate(pieces):
            out.append(self._make_chunk(
                doc_id=doc_id, main_title=main_title,
                section_title=section_title or f"part{i+1}", content=p,
                page=page, idx=i, section_path=section_path or main_title,
            ))
        return out
