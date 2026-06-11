"""
索引脚本 - 批量索引文档
用法:
    python scripts/index_files.py --dir ../company/          # 索引目录
    python scripts/index_files.py --file test.md           # 索引单个文件
    python scripts/index_files.py --delete test.pdf        # 删除索引
    python scripts/index_files.py --list                    # 列出已索引文件
"""
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def index_directory(dir_path):
    """索引目录"""
    from src.services.index_service import get_index_service

    service = get_index_service()
    print(f"📂 开始索引目录: {dir_path}")
    print()

    results = service.index_directory(dir_path, recursive=True)

    print()
    stats = service.get_stats()
    print(f"📊 当前总条目数: {stats.get('row_count', 0)}")


def index_file(file_path):
    """索引单个文件"""
    from src.services.index_service import get_index_service

    service = get_index_service()
    print(f"📄 开始索引: {file_path}")

    result = service.index_file(file_path)

    if result.get('success'):
        print(f"✅ 索引成功，生成 {result.get('chunk_count', 0)} 个分片")
    else:
        print(f"❌ 索引失败: {result.get('error', '未知错误')}")

    stats = service.get_stats()
    print(f"📊 当前总条目数: {stats.get('row_count', 0)}")


def delete_file(file_path):
    """删除文件的索引"""
    from src.services.index_service import get_index_service

    service = get_index_service()
    print(f"🗑️  删除索引: {file_path}")

    if service.delete_file(file_path):
        print("✅ 删除成功")
    else:
        print("⚠️  删除失败或文件未索引")


def list_files():
    """列出已索引的文件"""
    from src.services.index_service import get_index_service

    service = get_index_service()
    files = service.list_indexed_files()
    stats = service.get_stats()

    print(f"📊 总条目数: {stats.get('row_count', 0)}")
    print(f"📄 已索引文件 ({len(files)} 个):")
    print()

    if files:
        for f in files:
            print(f"  - {f.get('file_name', '?')}  ({f.get('file_type', '?')})")
    else:
        print("  暂无")


def main():
    parser = argparse.ArgumentParser(description="知识库索引工具")
    parser.add_argument("--dir", "-d", help="索引目录下的所有文件")
    parser.add_argument("--file", "-f", help="索引单个文件")
    parser.add_argument("--delete", help="删除指定文件的索引")
    parser.add_argument("--list", "-l", action="store_true", help="列出已索引的文件")

    args = parser.parse_args()

    if args.list:
        list_files()
    elif args.delete:
        delete_file(args.delete)
    elif args.dir:
        index_directory(args.dir)
    elif args.file:
        index_file(args.file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
