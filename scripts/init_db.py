#!/usr/bin/env python3
"""
初始化 Milvus Lite 向量数据库
用法:
    python scripts/init_db.py              # 初始化
    python scripts/init_db.py --drop       # 删除旧库重新创建
    python scripts/init_db.py --info       # 查看当前状态
"""
import sys
import argparse
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.storage.vector_store import get_vector_store
from config import milvus_config


def init_database(drop_if_exists=False):
    """初始化数据库"""
    print("=" * 60)
    print("初始化 Milvus Lite 向量数据库")
    print("=" * 60)

    print(f"\n数据库路径: {milvus_config.DB_PATH}")
    print(f"Collection: {milvus_config.COLLECTION_NAME}")
    print(f"向量维度: {milvus_config.VECTOR_DIM}")
    print(f"距离度量: {milvus_config.METRIC_TYPE}")

    store = get_vector_store()

    if drop_if_exists:
        print("\n⚠️  将删除旧 Collection，确定要继续吗？")
        print("   按 Ctrl+C 取消，按回车继续...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n已取消")
            return

    store.create_collection(drop_if_exists=drop_if_exists)

    # 显示状态
    stats = store.get_stats()
    print(f"\n📊 当前状态:")
    print(f"   条目数: {stats.get('row_count', 0)}")

    print("\n✅ 数据库初始化完成！")


def show_info():
    """显示数据库信息"""
    print("=" * 60)
    print("Milvus Lite 数据库信息")
    print("=" * 60)

    store = get_vector_store()

    if not store.has_collection():
        print(f"\n❌ Collection {milvus_config.COLLECTION_NAME} 不存在")
        print("   请先运行: python scripts/init_db.py")
        return

    stats = store.get_stats()
    print(f"\n📁 数据库路径: {milvus_config.DB_PATH}")
    print(f"📚 Collection: {milvus_config.COLLECTION_NAME}")
    print(f"📊 总条目数: {stats.get('row_count', 0)}")

    # 列出已索引的文件
    files = store.list_files()
    if files:
        print(f"\n📄 已索引的文件 ({len(files)} 个):")
        for f in files:
            print(f"   - {f.get('file_name', '未知')} ({f.get('file_type', '?')})")
    else:
        print("\n📄 暂无已索引文件")

    print()


def main():
    parser = argparse.ArgumentParser(description="初始化 Milvus Lite 向量数据库")
    parser.add_argument("--drop", action="store_true", help="删除旧 Collection 重新创建")
    parser.add_argument("--info", action="store_true", help="仅查看当前状态")
    args = parser.parse_args()

    if args.info:
        show_info()
    else:
        init_database(drop_if_exists=args.drop)


if __name__ == "__main__":
    main()
