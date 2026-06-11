"""
PDF 文档解析器
"""
import sys
import os
import re
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from .base_parser import BaseParser, DocumentChunk


class PDFParser(BaseParser):
    """PDF 文档解析器（基于 PyMuPDF）"""

    # 章节标题正则（国标风格：1 / 1.1 / 1.1.1）
    CHAPTER_PATTERN = re.compile(r'^\s*(\d+(\.\d+)*)\s+[^\d].*$')

    def __init__(self, chunk_size=500, chunk_overlap=100, extract_tables=True):
        super().__init__(chunk_size, chunk_overlap)
        self.extract_tables = extract_tables

    def parse(self, file_path: str) -> List[DocumentChunk]:
        """解析 PDF 文件"""
        import fitz  # PyMuPDF

        file_name = os.path.basename(file_path)
        file_hash = self.get_file_hash(file_path)

        doc = fitz.open(file_path)
        chunks = []
        chunk_idx = 0

        current_chapter = ""

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()

            if not page_text.strip():
                continue

            # 尝试识别章节标题
            chapter = self._detect_chapter(page_text, current_chapter)
            if chapter:
                current_chapter = chapter

            # 提取表格
            tables = []
            if self.extract_tables:
                tables = self._extract_tables(page, page_num)

            # 文本切片
            text_chunks = self.split_text(page_text)

            for tc in text_chunks:
                # 注入章节和页码上下文
                enriched = f"【文档】{file_name}\n【页码】第 {page_num + 1} 页"
                if current_chapter:
                    enriched += f"\n【章节】{current_chapter}"
                enriched += f"\n【内容】{tc}"

                chunk = DocumentChunk(
                    file_path=file_path,
                    file_name=file_name,
                    file_type='pdf',
                    file_hash=file_hash,
                    content=enriched,
                    content_type='text',
                    page_num=page_num + 1,
                    chapter_path=current_chapter,
                    chunk_index=chunk_idx,
                )
                chunks.append(chunk)
                chunk_idx += 1

            # 表格单独作为 chunk
            for table_content in tables:
                enriched = f"【文档】{file_name}\n【页码】第 {page_num + 1} 页"
                if current_chapter:
                    enriched += f"\n【章节】{current_chapter}"
                enriched += f"\n【表格】\n{table_content}"

                chunk = DocumentChunk(
                    file_path=file_path,
                    file_name=file_name,
                    file_type='pdf',
                    file_hash=file_hash,
                    content=enriched,
                    content_type='table',
                    page_num=page_num + 1,
                    chapter_path=current_chapter,
                    chunk_index=chunk_idx,
                )
                chunks.append(chunk)
                chunk_idx += 1

        doc.close()
        return chunks

    def _detect_chapter(self, page_text: str, current_chapter: str) -> str:
        """尝试检测页面中的章节标题"""
        lines = page_text.strip().split('\n')
        for line in lines[:10]:  # 只看前 10 行
            line = line.strip()
            if self.CHAPTER_PATTERN.match(line):
                # 提取章节编号
                match = re.match(r'^\s*(\d+(\.\d+)*)', line)
                if match:
                    chap_num = match.group(1)
                    # 如果是新的章节（级别比当前高或同级）
                    if current_chapter:
                        # 简单比较复杂，这里简化处理
                        return line
                    else:
                        return line
        return ""

    def _extract_tables(self, page, page_num: int) -> List[str]:
        """提取 PDF 表格（用 pdfplumber 效果更好，这里用简单实现）

        如需更好的表格提取，建议安装 pdfplumber:
            pip install pdfplumber

        这里提供基于 PyMuPDF 的简单实现
        """
        tables = []

        try:
            # 尝试用 PyMuPDF 的表格检测（需要较新版本）
            if hasattr(page, 'find_tables'):
                found_tables = page.find_tables()
                if found_tables:
                    for tab in found_tables:
                        try:
                            table_data = tab.extract()
                            if table_data and len(table_data) > 1:
                                # 转为 markdown 表格
                                md_table = self._table_to_markdown(table_data)
                                tables.append(md_table)
                        except Exception:
                            pass
        except Exception:
            pass

        # 如果 PyMuPDF 没找到表格，尝试用简单方法
        if not tables:
            # 简单规则：行内有多个制表符或多个空格分隔的列
            pass

        return tables

    def _table_to_markdown(self, table_data: List[List[str]]) -> str:
        """将表格数据转为 markdown 格式"""
        if not table_data:
            return ""

        # 过滤空行
        table_data = [row for row in table_data if any(cell and cell.strip() for cell in row)]

        if len(table_data) < 2:
            return ""

        # 找最大列数
        max_cols = max(len(row) for row in table_data)

        # 补全每行列数
        table_data = [list(row) + [''] * (max_cols - len(row)) for row in table_data]

        # 生成 markdown
        lines = []
        # 表头
        lines.append('| ' + ' | '.join(str(cell).strip() if cell else '' for cell in table_data[0]) + ' |')
        # 分隔线
        lines.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
        # 内容
        for row in table_data[1:]:
            lines.append('| ' + ' | '.join(str(cell).strip() if cell else '' for cell in row) + ' |')

        return '\n'.join(lines)
