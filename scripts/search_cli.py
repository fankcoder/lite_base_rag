#!/usr/bin/env python3
"""
命令行检索工具
用法:
    python scripts/search_cli.py "你的问题"
    python scripts/search_cli.py "你的问题" --top 5
    python scripts/search_cli.py "你的问题" --type text
"""
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def search(query, top_k=10, content_type=None):
    """搜索并打印结果"""
    from src.retriever.retriever import get_retriever

    retriever = get_retriever()

    print(f"\n🔍 查询: {query}")
    print(f"📊 Top-{top_k} 结果:")
    print("=" * 80)

    results = retriever.search(query, top_k=top_k, content_type=content_type)

    if not results:
        print("😕 未找到相关结果")
        return

    for i, r in enumerate(results, 1):
        print(f"\n【{i}】 相似度: {r['score']:.4f}")
        print(f"     来源: {r['file_name']} ", end='')
        if r.get('page_num'):
            print(f"(第 {r['page_num']} 页)", end='')
        if r.get('slide_num'):
            print(f"(PPT 第 {r['slide_num']} 页)", end='')
        if r.get('chapter_path'):
            print(f"\n     章节: {r['chapter_path']}", end='')
        print(f"\n     类型: {r.get('content_type', 'text')}")
        print()

        # 内容预览（截断）
        content = r.get('content', '')
        # 如果有【内容】标记，只显示后面的部分
        if '【内容】' in content:
            content = content.split('【内容】')[-1].strip()

        if len(content) > 300:
            content = content[:300] + '...'

        print(f"     {content}")
        print()

    print("=" * 80)
    print(f"共 {len(results)} 条结果")


def main():
    parser = argparse.ArgumentParser(description="知识库检索工具")
    parser.add_argument("query", help="查询文本")
    parser.add_argument("--top", "-k", type=int, default=10, help="返回数量")
    parser.add_argument("--type", "-t", choices=["text", "table", "image"],
                        help="按内容类型过滤")

    args = parser.parse_args()
    search(args.query, top_k=args.top, content_type=args.type)


if __name__ == "__main__":
    main()
