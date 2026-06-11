"""
Markdown 文档解析器
"""
import re
import sys
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from .base_parser import BaseParser, DocumentChunk


class MDParser(BaseParser):
    """Markdown 文档解析器"""

    # 标题正则
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+?)\s*#*\s*$', re.MULTILINE)
    # 表格正则（简单版）
    TABLE_PATTERN = re.compile(r'^\|.*\|$\n^\|[-:\s|]+\|$\n(?:^\|.*\|$\n?)*', re.MULTILINE)

    def parse(self, file_path: str) -> List[DocumentChunk]:
        """解析 Markdown 文件"""
        import os

        file_name = os.path.basename(file_path)
        file_hash = self.get_file_hash(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        chunks = []

        # 1. 提取所有标题位置
        headings = self._extract_headings(content)

        # 2. 按标题分段
        sections = self._split_by_headings(content, headings)

        # 3. 对每个 section 切片
        chunk_idx = 0
        for section in sections:
            section_text = section['text']
            chapter_path = section['chapter']

            # 检查是否包含表格
            tables = self._extract_tables(section_text)

            # 如果表格比较大，单独作为 chunk
            if tables:
                # 移除表格后的纯文本
                text_without_tables = self.TABLE_PATTERN.sub('', section_text).strip()

                # 纯文本切片
                if text_without_tables:
                    text_chunks = self.split_text(text_without_tables)
                    for tc in text_chunks:
                        chunk = DocumentChunk(
                            file_path=file_path,
                            file_name=file_name,
                            file_type='md',
                            file_hash=file_hash,
                            content=tc,
                            content_type='text',
                            chapter_path=chapter_path,
                            chunk_index=chunk_idx,
                        )
                        chunks.append(chunk)
                        chunk_idx += 1

                # 表格单独作为 chunk
                for table in tables:
                    chunk = DocumentChunk(
                        file_path=file_path,
                        file_name=file_name,
                        file_type='md',
                        file_hash=file_hash,
                        content=table,
                        content_type='table',
                        chapter_path=chapter_path,
                        chunk_index=chunk_idx,
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
            else:
                # 没有表格，直接切片
                text_chunks = self.split_text(section_text)
                for tc in text_chunks:
                    # 注入章节上下文
                    enriched_content = f"【章节】{chapter_path}\n【内容】{tc}" if chapter_path else tc

                    chunk = DocumentChunk(
                        file_path=file_path,
                        file_name=file_name,
                        file_type='md',
                        file_hash=file_hash,
                        content=enriched_content,
                        content_type='text',
                        chapter_path=chapter_path,
                        chunk_index=chunk_idx,
                    )
                    chunks.append(chunk)
                    chunk_idx += 1

        return chunks

    def _extract_headings(self, content: str) -> List[dict]:
        """提取所有标题"""
        headings = []
        for match in self.HEADING_PATTERN.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            pos = match.start()
            headings.append({
                'level': level,
                'text': text,
                'position': pos,
            })
        return headings

    def _split_by_headings(self, content: str, headings: List[dict]) -> List[dict]:
        """按标题拆分内容"""
        if not headings:
            return [{'text': content, 'chapter': ''}]

        sections = []
        heading_stack = []  # 维护当前章节路径

        for i, heading in enumerate(headings):
            # 更新章节栈
            while heading_stack and heading_stack[-1]['level'] >= heading['level']:
                heading_stack.pop()
            heading_stack.append(heading)

            # 计算当前章节的内容范围
            start_pos = heading['position']
            end_pos = headings[i + 1]['position'] if i + 1 < len(headings) else len(content)

            # 跳过标题行本身
            section_text = content[start_pos:end_pos].strip()

            # 章节路径
            chapter_path = ' > '.join([h['text'] for h in heading_stack])

            sections.append({
                'text': section_text,
                'chapter': chapter_path,
            })

        return sections

    def _extract_tables(self, text: str) -> List[str]:
        """提取 markdown 表格"""
        tables = []
        for match in self.TABLE_PATTERN.finditer(text):
            tables.append(match.group(0).strip())
        return tables
