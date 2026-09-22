"""
长对话自动摘要中间件。

核心机制：
- SQLite 负责“持久化”；
- SummarizationMiddleware 负责“压缩过长上下文”。

二者组合后：
    完整聊天不断增长
        -> 达到阈值
        -> 较老消息被 LLM 摘要
        -> 最近消息保持原文
        -> 摘要后的 state 继续写入 SQLite

因此程序重启后仍然可以恢复压缩后的有效记忆。
"""

from langchain.agents.middleware import SummarizationMiddleware

from agent.prompts import MEMORY_SUMMARY_PROMPT
from config import settings
from models.model_factory import create_summary_model


def create_conversation_summary_middleware() -> SummarizationMiddleware:
    """创建 Agent 使用的自动摘要 Middleware。"""

    summary_model = create_summary_model()

    # list 表示 OR：
    # tokens >= threshold 或 messages >= threshold，任意一个满足就触发摘要。
    trigger = [
        ("tokens", settings.summary_trigger_tokens),
        ("messages", settings.summary_trigger_messages),
    ]

    return SummarizationMiddleware(
        model=summary_model,
        trigger=trigger,
        # 摘要完成后，最近 N 条 messages 仍然保留原文，
        # 避免刚刚发生的 Tool Call / 用户问题被过度压缩。
        keep=("messages", settings.summary_keep_messages),
        summary_prompt=MEMORY_SUMMARY_PROMPT,
        trim_tokens_to_summarize=settings.summary_trim_tokens,
    )
