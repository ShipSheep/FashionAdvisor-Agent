"""
项目统一配置文件。

项目配置主要包含以下两类检索与搜索能力：
1. Local RAG relevance gate：判断本地知识库是否真的命中了“有用知识”；
2. Tavily Web Search：当本地知识不足或问题本身具有明显时效性时，作为公网搜索能力。

仍然坚持一个原则：
    配置与业务代码分离。

以后更换模型、调整 RAG 阈值、修改 Tavily 搜索深度时，
只需要改 .env / config.py，不需要改 Agent 主流程。
"""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    """把环境变量中的 true/false/1/0 转换为 Python bool。"""

    raw = os.getenv(name)
    if raw is None:
        return default

    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


@dataclass(frozen=True)
class Settings:
    """整个项目的只读配置对象。"""

    # ================================================================
    # 1. 主 Agent Chat Model
    # ================================================================
    # Qwen 通过 DashScope OpenAI-compatible API 调用。
    # Agent 层始终只依赖 init_chat_model，不直接依赖某个模型类。
    model_provider: str = os.getenv("MODEL_PROVIDER", "openai")
    model_name: str = os.getenv("MODEL_NAME", "qwen3-max")
    model_api_key: str = os.getenv("MODEL_API_KEY", "")
    model_base_url: str = os.getenv(
        "MODEL_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model_temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
    model_max_tokens: int = int(os.getenv("MODEL_MAX_TOKENS", "2000"))
    model_max_retries: int = int(os.getenv("MODEL_MAX_RETRIES", "2"))

    # ================================================================
    # 2. 长对话摘要模型
    # ================================================================
    # 可以和主模型相同，也可以单独切成便宜模型。
    summary_model_provider: str = os.getenv(
        "SUMMARY_MODEL_PROVIDER",
        os.getenv("MODEL_PROVIDER", "openai"),
    )
    summary_model_name: str = os.getenv(
        "SUMMARY_MODEL_NAME",
        os.getenv("MODEL_NAME", "qwen3-max"),
    )
    summary_model_api_key: str = os.getenv(
        "SUMMARY_MODEL_API_KEY",
        os.getenv("MODEL_API_KEY", ""),
    )
    summary_model_base_url: str = os.getenv(
        "SUMMARY_MODEL_BASE_URL",
        os.getenv(
            "MODEL_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )
    summary_model_temperature: float = float(
        os.getenv("SUMMARY_MODEL_TEMPERATURE", "0.0")
    )
    summary_model_max_tokens: int = int(
        os.getenv("SUMMARY_MODEL_MAX_TOKENS", "1200")
    )

    # ================================================================
    # 3. Embedding
    # ================================================================
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "text-embedding-v4",
    )

    # ================================================================
    # 4. Chroma / RAG
    # ================================================================
    chroma_directory: Path = BASE_DIR / os.getenv("CHROMA_DIR", "chroma_db")
    chroma_collection_name: str = os.getenv(
        "CHROMA_COLLECTION",
        "shopping_agent_kb_v4",
    )

    # MMR：先取 fetch_k 个候选，再挑 top_k 个互补的 anchor chunks。
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    retrieval_fetch_k: int = int(os.getenv("RETRIEVAL_FETCH_K", "10"))
    retrieval_mmr_lambda: float = float(
        os.getenv("RETRIEVAL_MMR_LAMBDA", "0.65")
    )

    # 命中 anchor 后，前后各补几个邻居 chunk。
    retrieval_neighbor_window: int = int(
        os.getenv("RETRIEVAL_NEIGHBOR_WINDOW", "1")
    )

    # ================================================================
    # 5. Local RAG Relevance Gate
    # ================================================================
    # 为什么需要 gate？
    # 向量数据库即使面对“完全不相关的问题”，往往也能硬返回 Top-K。
    # 因此不能只用 `if not documents` 判断“知识库是否有答案”。
    #
    # 这里先做一次 relevance search，观察最高相关度。
    # 如果低于阈值，就不把那些“勉强最相似”的文档当作可靠知识。
    retrieval_gate_k: int = int(os.getenv("RETRIEVAL_GATE_K", "3"))
    retrieval_relevance_threshold: float = float(
        os.getenv("RETRIEVAL_RELEVANCE_THRESHOLD", "0.45")
    )

    # ================================================================
    # 6. Chunk 切分
    # ================================================================
    # chunk_size 是目标上限，不是固定每 800 字硬切。
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    chunking_version: str = os.getenv(
        "CHUNKING_VERSION",
        "structure_recursive_v1",
    )

    # ================================================================
    # 7. Tavily Web Search
    # ================================================================
    tavily_enabled: bool = _env_bool("TAVILY_ENABLED", True)
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    tavily_max_results: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    tavily_topic: str = os.getenv("TAVILY_TOPIC", "general")
    tavily_search_depth: str = os.getenv(
        "TAVILY_SEARCH_DEPTH",
        "advanced",
    )

    # 默认不要求 Tavily 自己再生成 answer，也不抓 raw HTML。
    # 我们希望：Tavily 负责 Search，Agent 负责最终推理与回答。
    tavily_include_answer: bool = _env_bool(
        "TAVILY_INCLUDE_ANSWER",
        False,
    )
    tavily_include_raw_content: bool = _env_bool(
        "TAVILY_INCLUDE_RAW_CONTENT",
        False,
    )

    # ================================================================
    # 8. SQLite Agent Memory
    # ================================================================
    sqlite_db_path: Path = BASE_DIR / os.getenv(
        "SQLITE_DB_PATH",
        "data/agent_memory.sqlite3",
    )

    # ================================================================
    # 9. Conversation Summarization
    # ================================================================
    summary_trigger_tokens: int = int(
        os.getenv("SUMMARY_TRIGGER_TOKENS", "6000")
    )
    summary_trigger_messages: int = int(
        os.getenv("SUMMARY_TRIGGER_MESSAGES", "30")
    )
    summary_keep_messages: int = int(
        os.getenv("SUMMARY_KEEP_MESSAGES", "10")
    )
    summary_trim_tokens: int = int(
        os.getenv("SUMMARY_TRIM_TOKENS", "5000")
    )

    # ================================================================
    # 10. Data files
    # ================================================================
    data_directory: Path = BASE_DIR / "data"
    size_chart_path: Path = BASE_DIR / "data" / "size_chart.json"
    ingested_registry_path: Path = (
        BASE_DIR / "data" / "ingested_documents_v4.txt"
    )


settings = Settings()
