"""
工具函数模块
"""
import hashlib
import time
import re


def get_file_hash(file_path: str) -> str:
    """计算文件 MD5 哈希"""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def get_time_str(ts: float = None) -> str:
    """格式化时间字符串"""
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def clean_text(text: str) -> str:
    """清理文本"""
    if not text:
        return ""
    # 去除多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去除多余空格
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def truncate_text(text: str, max_len: int = 200, suffix: str = "...") -> str:
    """截断文本"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + suffix


def count_tokens(text: str) -> int:
    """
    粗略估算 token 数
    中文按 1 字 = 1.3 tokens，英文按 1 词 = 1.3 tokens
    """
    if not text:
        return 0

    # 统计中文字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 统计英文单词
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    # 数字
    numbers = len(re.findall(r'\d+', text))

    total = chinese_chars * 1.3 + english_words * 1.3 + numbers * 1.3
    return int(total)
