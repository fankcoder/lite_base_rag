#!/usr/bin/env python3
"""
企业知识库快速检索脚本（OpenClaw 调用专用）
更简洁的输出，适合程序解析

用法:
    python rag_query.py "你的问题"
    python rag_query.py "你的问题" --top 5
    python rag_query.py "你的问题" --json  # JSON 格式输出
"""
import sys
import os
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 配置 HuggingFace 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def search(query: str, top_k: int = 5, output_json: bool = False):
    """检索知识库"""
    from src.retriever import get_retriever

    retriever = get_retriever()
    results = retriever.search(query, top_k=top_k)

    if output_json:
        # JSON 格式输出
        output = {
            "query": query,
            "total": len(results),
            "results": []
        }
        for r in results:
            content = r.get('content', '')
            if '【内容】' in content:
                content = content.split('【内容】', 1)[-1].strip()

            output["results"].append({
                "score": round(r.get('score', 0), 4),
                "file_name": r.get('file_name', ''),
                "page_num": r.get('page_num', 0),
                "slide_num": r.get('slide_num', 0),
                "chapter": r.get('chapter_path', ''),
                "content_type": r.get('content_type', 'text'),
                "content": content[:1000],  # 截断
            })
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        # 人类可读格式
        print(f"🔍 查询: {query}\n")
        print(f"📊 找到 {len(results)} 条相关结果:\n")

        for i, r in enumerate(results, 1):
            score = r.get('score', 0)
            file_name = r.get('file_name', '未知')
            content = r.get('content', '')
            chapter = r.get('chapter_path', '')

            if '【内容】' in content:
                content = content.split('【内容】', 1)[-1].strip()

            if len(content) > 400:
                content = content[:400] + "..."

            print(f"【{i}】相似度 {score:.4f}")
            print(f"📄 {file_name}", end='')
            if r.get('page_num'):
                print(f" (第 {r['page_num']} 页)", end='')
            if r.get('slide_num'):
                print(f" (PPT第 {r['slide_num']} 页)", end='')
            print()
            if chapter:
                print(f"📑 {chapter}")
            print(f"📝 {content}")
            print()

    return results


def main():
    parser = argparse.ArgumentParser(description="企业知识库检索")
    parser.add_argument("query", help="查询问题")
    parser.add_argument("--top", "-k", type=int, default=5, help="返回结果数量")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--type", "-t", choices=["text", "table"], help="按内容类型过滤")

    args = parser.parse_args()
    search(args.query, top_k=args.top, output_json=args.json)


if __name__ == "__main__":
    main()
