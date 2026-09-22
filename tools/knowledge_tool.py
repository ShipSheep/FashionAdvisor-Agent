"""
本地 RAG Tool。

当前实现的关键区别：
Tool 不再只返回“检索到的文本”，还会明确告诉 Agent：
    LOCAL_KB_STATUS=SUFFICIENT
或：
    LOCAL_KB_STATUS=INSUFFICIENT

这让 Agent 能够在本地证据不足时继续调用 Tavily Web Search。
"""

from pydantic import BaseModel, Field
from langchain.tools import tool

from rag.retriever import KnowledgeRetriever


class KnowledgeSearchInput(BaseModel):
    """search_knowledge_base 的结构化输入 Schema。"""

    query: str = Field(
        ...,
        description=(
            "要在本地服装知识库中检索的完整独立问题。"
            "如果用户只说‘那羊绒呢？’，请结合历史改写成"
            "‘羊绒衣物应该如何洗涤和养护？’。"
        ),
    )


def create_knowledge_search_tool(
    retriever: KnowledgeRetriever,
):
    """通过依赖注入创建本地知识库 Tool。"""

    @tool(
        "search_knowledge_base",
        args_schema=KnowledgeSearchInput,
        description=(
            "查询本地服装知识库。"
            "适用于颜色搭配、场合建议、材质、洗涤、养护等本地静态知识。"
            "工具会进行相关性 Gate，并返回 LOCAL_KB_STATUS。"
            "如果状态为 INSUFFICIENT，且问题属于公开互联网知识，"
            "应继续考虑调用 web_search。"
            "明确尺码问题必须优先使用 recommend_size。"
        ),
    )
    def search_knowledge_base(query: str) -> str:
        result = retriever.search(query)
        return retriever.format_result(result)

    return search_knowledge_base
