#!/usr/bin/env python3
"""
企业知识库 RAG - MCP Server
通过 MCP 协议暴露知识库检索工具，供 OpenClaw 调用

用法:
    python mcp_server.py         # stdio 模式
    mcp-inspect python mcp_server.py
"""
import sys
import os
from pathlib import Path

# 确保项目目录在 sys.path 中
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 配置 HuggingFace 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP("企业知识库 RAG", description="企业知识库向量检索工具")


@mcp.tool(
    name="search_knowledge_base",
    description="检索企业知识库，返回相关文档片段和来源。用于回答公司产品、服务、规范、案例、标准等内部问题。"
)
def search_knowledge_base(
    query: str,
    top_k: int = 5,
    content_type: str = None,
) -> str:
    """
    检索企业知识库

    Args:
        query: 查询问题，用中文描述
        top_k: 返回结果数量，默认 5 条
        content_type: 内容类型过滤，可选 text / table，默认不过滤

    Returns:
        检索结果字符串，包含相似度、来源、内容
    """
    try:
        from src.retriever import get_retriever

        retriever = get_retriever()
        results = retriever.search(query, top_k=top_k, content_type=content_type)

        if not results:
            return "未找到相关内容。建议换一种问法，或确认该问题是否在知识库覆盖范围内。"

        # 格式化输出
        output = f"知识库检索结果（共 {len(results)} 条，按相似度排序）:\n\n"
        for i, r in enumerate(results, 1):
            score = r.get('score', 0)
            file_name = r.get('file_name', '未知文件')
            content = r.get('content', '')
            chapter = r.get('chapter_path', '')
            content_type = r.get('content_type', 'text')

            # 提取纯内容部分（去掉【文档】【章节】等前缀）
            if '【内容】' in content:
                content = content.split('【内容】', 1)[-1].strip()

            # 截断太长的内容
            if len(content) > 500:
                content = content[:500] + "..."

            output += f"【结果 {i}】相似度: {score:.4f}\n"
            output += f"📄 来源: {file_name}"
            if r.get('page_num'):
                output += f" (第 {r['page_num']} 页)"
            if r.get('slide_num'):
                output += f" (PPT第 {r['slide_num']} 页)"
            if chapter:
                output += f"\n📑 章节: {chapter}"
            output += f"\n🏷️ 类型: {content_type}"
            output += f"\n📝 内容:\n{content}\n"
            output += "\n" + "-" * 50 + "\n\n"

        output += "\n💡 提示: 请基于以上检索内容回答用户问题，不得编造信息。如果没有相关内容，请明确告知用户。"

        return output

    except Exception as e:
        import traceback
        return f"检索出错: {str(e)}\n\n{traceback.format_exc()}"


@mcp.tool(
    name="list_indexed_documents",
    description="列出知识库中已索引的所有文档清单"
)
def list_indexed_documents() -> str:
    """
    列出已索引的文档

    Returns:
        文档列表字符串
    """
    try:
        from src.storage import get_vector_store

        store = get_vector_store()
        files = store.list_files()

        if not files:
            return "知识库中没有已索引的文档。"

        output = f"知识库已索引文档（共 {len(files)} 个）:\n\n"
        for f in files:
            output += f"📄 {f.get('file_name', '未知')}"
            output += f"  [{f.get('file_type', '?')}]"
            output += "\n"

        output += f"\n总计 {len(files)} 个文档"
        return output

    except Exception as e:
        return f"获取文档列表出错: {str(e)}"


if __name__ == "__main__":
    # stdio 模式运行
    mcp.run()
