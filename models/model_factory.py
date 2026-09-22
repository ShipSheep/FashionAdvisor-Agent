"""
统一模型工厂。

核心要求：
    from langchain.chat_models import init_chat_model

Agent 代码不直接依赖 ChatOpenAI / ChatTongyi / ChatAnthropic 等具体类。
只要 provider 集成包安装正确，就可以通过 .env 切换模型。
"""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings

from config import settings


def _build_chat_model(
    *,
    provider: str,
    model_name: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 2,
) -> BaseChatModel:
    """内部统一构造函数，主模型和摘要模型都复用这里。"""

    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
    }

    # 某些 provider 不需要 base_url，因此只有非空时才传入。
    if api_key:
        kwargs["api_key"] = api_key

    if base_url:
        kwargs["base_url"] = base_url

    return init_chat_model(
        model=model_name,
        model_provider=provider,
        **kwargs,
    )


def create_chat_model() -> BaseChatModel:
    """创建主 Agent 使用的 Chat Model。"""

    return _build_chat_model(
        provider=settings.model_provider,
        model_name=settings.model_name,
        api_key=settings.model_api_key,
        base_url=settings.model_base_url,
        temperature=settings.model_temperature,
        max_tokens=settings.model_max_tokens,
        max_retries=settings.model_max_retries,
    )


def create_summary_model() -> BaseChatModel:
    """
    创建历史摘要模型。

    为什么单独创建？
    - 现在可以和 Agent 使用同一个 Qwen；
    - 以后也可以把 SUMMARY_MODEL_NAME 改成更便宜的小模型；
    - Agent 业务代码完全不用改。
    """

    return _build_chat_model(
        provider=settings.summary_model_provider,
        model_name=settings.summary_model_name,
        api_key=settings.summary_model_api_key,
        base_url=settings.summary_model_base_url,
        temperature=settings.summary_model_temperature,
        max_tokens=settings.summary_model_max_tokens,
        max_retries=settings.model_max_retries,
    )


def create_embedding_model() -> DashScopeEmbeddings:
    """
    创建 RAG 使用的 Embedding 模型。

    Embedding 与 Agent Chat Model 是两个独立组件。
    即使未来 Agent 换成 Claude，Embedding 仍然可以继续使用 DashScope。
    """

    return DashScopeEmbeddings(
        model=settings.embedding_model_name,
    )
