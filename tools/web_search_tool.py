"""
Tavily 联网搜索 Tool。

为什么不把 Tavily 写死在 search_knowledge_base 内部？
------------------------------------------------------
因为我们希望保持工具职责单一：
- search_knowledge_base：只查本地私有/静态知识；
- web_search：只查公开互联网。

是否从本地知识切换到公网知识，由 Agent 决策。
这样实时问题可以直接联网，本地问题也不会无意义地产生公网调用。
"""

import json
from datetime import datetime
from typing import Any, Literal

from langchain.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from config import settings


class WebSearchInput(BaseModel):
    """web_search 的输入 Schema。"""

    query: str = Field(
        ...,
        description=(
            "用于互联网搜索的完整自然语言问题。"
            "应包含用户真正需要的关键信息，不要只传‘这个呢’之类省略问句。"
        ),
    )

    time_range: Literal[
        "day",
        "week",
        "month",
        "year",
    ] | None = Field(
        default=None,
        description=(
            "可选时间范围。只有用户强调最新、近期、今年、当前趋势时才设置。"
        ),
    )


def _normalize_tavily_response(response: Any) -> dict[str, Any]:
    """把 Tavily 的不同返回形态尽量统一成 dict。"""

    if isinstance(response, dict):
        return response

    # 如果未来版本返回 ToolMessage，可尝试读取其 content。
    content = getattr(response, "content", None)
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"results": [], "raw_text": content}

    if isinstance(response, str):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"results": [], "raw_text": response}

    return {"results": []}


def create_web_search_tool():
    """
    创建 Tavily Tool。

    当 TAVILY_ENABLED=true 时如果没有 API Key，直接报清晰错误，
    避免 Agent 看似拥有联网能力但真正调用时才莫名失败。
    """

    if not settings.tavily_api_key:
        raise ValueError(
            "TAVILY_ENABLED=true，但没有配置 TAVILY_API_KEY。"
            "请在 .env 中填写 Tavily API Key，或把 TAVILY_ENABLED=false。"
        )

    # Tavily 官方 LangChain 集成。
    tavily_client = TavilySearch(
        max_results=settings.tavily_max_results,
        topic=settings.tavily_topic,
        search_depth=settings.tavily_search_depth,
        include_answer=settings.tavily_include_answer,
        include_raw_content=settings.tavily_include_raw_content,
        include_images=False,
    )

    @tool(
        "web_search",
        args_schema=WebSearchInput,
        description=(
            "使用 Tavily 搜索公开互联网。"
            "适用于：本地知识库相关性不足、用户明确要求联网、"
            "或问题具有明显实时性，例如最新趋势、今年流行、近期资讯。"
            "不要用它替代 recommend_size 的内部尺码规则；"
            "也不要用公网搜索结果冒充本系统内部私有知识。"
        ),
    )
    def web_search(
        query: str,
        time_range: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "query": query,
        }

        if time_range:
            payload["time_range"] = time_range

        raw_response = tavily_client.invoke(payload)
        response = _normalize_tavily_response(raw_response)

        results = response.get("results", [])

        if not isinstance(results, list) or not results:
            return (
                "WEB_SEARCH_STATUS=NO_RESULTS\n"
                "Tavily 没有返回可用的互联网搜索结果。"
            )

        blocks: list[str] = [
            "WEB_SEARCH_STATUS=SUCCESS",
            f"WEB_SEARCH_QUERY={query}",
            f"WEB_SEARCH_TIME={datetime.now().isoformat(timespec='seconds')}",
            "说明：以下内容来自公开互联网，不代表本地知识库内部规则。",
        ]

        for index, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue

            title = str(item.get("title", "未命名网页"))
            url = str(item.get("url", ""))
            content = str(item.get("content", ""))
            score = item.get("score")

            score_text = "N/A"
            if isinstance(score, (int, float)):
                score_text = f"{float(score):.4f}"

            blocks.append(
                "\n".join(
                    [
                        f"[联网资料 {index}]",
                        f"标题：{title}",
                        f"URL：{url or '-'}",
                        f"Tavily Score：{score_text}",
                        "摘要：",
                        content,
                    ]
                )
            )

        return "\n\n".join(blocks)

    return web_search
