# companyrag - 企业知识库 RAG 系统

轻量级企业知识库多模态 RAG 系统，支持 Markdown / PDF / PPT 等多种格式文档的向量化存储与智能检索。

## ✨ 特性

- 📄 **多格式支持**: Markdown、PDF、PPTX 文档解析
- 🧠 **中文优化**: 基于 bge 系列 embedding 模型，中文效果好
- 🔍 **多路召回**: 向量检索 + BM25 关键词 + Rerank 精排
- 💾 **轻量部署**: Milvus Lite 单文件向量库，零配置
- 🚀 **API 服务**: FastAPI 提供 REST 接口，支持流式问答
- 🔄 **增量更新**: 基于文件哈希的自动增量索引
- 📊 **效果监控**: 内置效果评估与幻觉检测工具

## 🏗️ 技术栈

| 层级 | 技术 |
|------|------|
| 文档解析 | PyMuPDF + python-pptx + markdown |
| Embedding | bge-small-zh-v1.5 (可升级 bge-m3) |
| Rerank | bge-reranker-base |
| 向量数据库 | Milvus Lite |
| 服务框架 | FastAPI + Uvicorn |
| LLM 生成 | 外部 API + 本地模型兜底 |

## 📁 项目结构

```
companyrag/
├── README.md                # 本文件
├── requirements.txt       # Python 依赖
├── .env.example     # 环境变量模板
├── config.py          # 配置文件
├── scripts/          # 脚本工具
│   ├── init_db.py      # 初始化数据库
│   ├── index_files.py  # 批量索引文件
│   ├── search_cli.py    # 命令行检索
│   └── evaluate.py     # 效果评估
├── src/              # 源代码
│   ├── api/          # API 接口
│   ├── parser/       # 文档解析
│   ├── embedding/    # 向量化
│   ├── retriever/    # 检索
│   ├── generator/    # LLM 生成
│   ├── storage/      # 存储层
│   ├── services/     # 业务逻辑
│   └── utils/        # 工具函数
├── data/             # 数据目录
│   ├── milvus/       # Milvus Lite 数据库
│   ├── cache/        # 模型缓存
│   ├── processed/    # 解析后的中间文件
│   └── uploads/      # 上传文件
├── tests/            # 测试
└── examples/         # 示例
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
```

### 2. 初始化向量数据库

```bash
python scripts/init_db.py
```

### 3. 索引知识库文件

```bash
# 索引指定目录下的所有文件
python scripts/index_files.py --dir ../company/
```

### 4. 命令行检索测试

```bash
python scripts/search_cli.py "公司的核心产品有哪些？
```

### 5. 启动 API 服务

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 http://localhost:8000/docs 查看 API 文档。

## 📖 使用指南

详细设计文档见 `../rag_plan/ 目录。

## 🔧 配置说明

主要配置在 `.env` 文件中：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `EMBEDDING_MODEL | BAAI/bge-small-zh-v1.5 | Embedding 模型 |
| `EMBEDDING_DEVICE | cpu | 运行设备 cpu/cuda |
| `MILVUS_DB_PATH | ./data/milvus/milvus.db | Milvus Lite 数据库路径 |
| `CHUNK_SIZE | 500 | 文本分片大小（字 |
| `CHUNK_OVERLAP | 100 | 文本分片重叠大小 |
| `TOP_K | 10 | 检索返回数量 |
| `RERANK_ENABLED | false | 是否启用 Rerank |
| `LLM_API_KEY | - | LLM API Key |
| `LLM_MODEL | deepseek-chat | LLM 模型名称 |

## 📊 效果评估

```bash
python scripts/evaluate.py --testset tests/test_queries.json
```

## 🤝 开发说明

### 添加新的文档解析器：

1. 在 `src/parser/` 中继承 `BaseParser`
2. 实现 `parse()` 方法
3. 在 `src/parser/__init__.py` 中注册

## 📝 License

MIT
