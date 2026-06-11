"""
配置文件
使用 python-dotenv 从 .env 文件加载配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 加载 .env 文件
load_dotenv(BASE_DIR / ".env")


# ==================== Embedding 配置 ====================
class EmbeddingConfig:
    MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    NORMALIZE_EMBEDDINGS = os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
    MAX_SEQ_LENGTH = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "512"))
    BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    CACHE_DIR = BASE_DIR / "data" / "cache"


# ==================== Milvus 配置 ====================
class MilvusConfig:
    TYPE = os.getenv("MILVUS_TYPE", "lite")  # lite / server
    DB_PATH = os.getenv("MILVUS_DB_PATH", str(BASE_DIR / "data" / "milvus" / "milvus.db"))
    COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "knowledge_base")
    VECTOR_DIM = int(os.getenv("MILVUS_VECTOR_DIM", "512"))
    METRIC_TYPE = os.getenv("MILVUS_METRIC", "COSINE")
    INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "FLAT")


# ==================== 文档解析配置 ====================
class ParserConfig:
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "50"))
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    SUPPORTED_FORMATS = [".md", ".pdf", ".pptx", ".docx", ".txt"]


# ==================== 检索配置 ====================
class RetrieverConfig:
    TOP_K = int(os.getenv("RETRIEVE_TOP_K", "10"))
    RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
    RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
    BM25_ENABLED = os.getenv("BM25_ENABLED", "true").lower() == "true"
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))


# ==================== LLM 配置 ====================
class LLMConfig:
    PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")  # deepseek / qwen / openai / local
    API_KEY = os.getenv("LLM_API_KEY", "")
    BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    STREAMING = os.getenv("LLM_STREAMING", "true").lower() == "true"


# ==================== 服务配置 ====================
class ServerConfig:
    HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    PORT = int(os.getenv("SERVER_PORT", "8000"))
    WORKERS = int(os.getenv("SERVER_WORKERS", "1"))
    API_KEY = os.getenv("API_KEY", "")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


# ==================== 知识库配置 ====================
class KnowledgeConfig:
    DIRECTORY = os.getenv("KNOWLEDGE_DIR", str(BASE_DIR / "data" / "uploads"))
    AUTO_INDEX = os.getenv("AUTO_INDEX", "false").lower() == "true"
    SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "3600"))  # 秒


# 导出配置实例
embedding_config = EmbeddingConfig()
milvus_config = MilvusConfig()
parser_config = ParserConfig()
retriever_config = RetrieverConfig()
llm_config = LLMConfig()
server_config = ServerConfig()
knowledge_config = KnowledgeConfig()
