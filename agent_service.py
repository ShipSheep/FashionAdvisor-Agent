"""
单 Agent 核心服务。

最终架构：

User
  -> create_agent
       |- init_chat_model 主模型
       |- Tools
       |    |- search_knowledge_base  本地 RAG + relevance gate
       |    |- web_search             Tavily 公网搜索
       |    |- recommend_size         结构化尺码规则
       |    `- convert_weight         确定性单位换算
       |- SummarizationMiddleware
       `- SqliteSaver

仍然只有一个 Agent，没有 Multi-Agent。
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agent.prompts import SYSTEM_PROMPT
from config import settings
from memory.sqlite_memory import SQLiteMemoryManager
from memory.summary_middleware import (
    create_conversation_summary_middleware,
)
from models.model_factory import create_chat_model
from rag.retriever import KnowledgeRetriever
from tools.knowledge_tool import create_knowledge_search_tool
from tools.size_tool import recommend_size
from tools.web_search_tool import create_web_search_tool
from tools.weight_tool import convert_weight


class FashionAdvisorAgentService:
    """项目对外暴露的单 Agent 服务。"""

    def __init__(self, debug: bool = False) -> None:
        # --------------------------------------------------------------
        # 1. 主模型
        # --------------------------------------------------------------
        # 所有 Chat Model 统一通过 init_chat_model 构建。
        self.model = create_chat_model()

        # --------------------------------------------------------------
        # 2. 本地 RAG Retriever -> Tool
        # --------------------------------------------------------------
        self.knowledge_retriever = KnowledgeRetriever()
        self.knowledge_tool = create_knowledge_search_tool(
            self.knowledge_retriever
        )

        # --------------------------------------------------------------
        # 3. 注册基础 Tools
        # --------------------------------------------------------------
        self.tools = [
            self.knowledge_tool,
            recommend_size,
            convert_weight,
        ]

        # --------------------------------------------------------------
        # 4. 可选 Tavily Web Search
        # --------------------------------------------------------------
        # 当 TAVILY_ENABLED=false 时，项目仍然可以作为纯本地 Agent 运行。
        # 当开启时，需要配置 TAVILY_API_KEY。
        self.web_search_tool = None

        if settings.tavily_enabled:
            self.web_search_tool = create_web_search_tool()
            self.tools.append(self.web_search_tool)

        # --------------------------------------------------------------
        # 5. SQLite 持久化
        # --------------------------------------------------------------
        self.memory_manager = SQLiteMemoryManager()

        # --------------------------------------------------------------
        # 6. 自动摘要 Middleware
        # --------------------------------------------------------------
        self.summary_middleware = (
            create_conversation_summary_middleware()
        )

        # --------------------------------------------------------------
        # 7. 创建 LangChain Agent
        # --------------------------------------------------------------
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=[
                self.summary_middleware,
            ],
            checkpointer=self.memory_manager.checkpointer,
            debug=debug,
            name="fashion_advisor_agent",
        )

    @staticmethod
    def _build_config(thread_id: str) -> dict[str, Any]:
        """构造 LangGraph thread 配置。"""

        return {
            "configurable": {
                "thread_id": thread_id,
            }
        }

    def invoke(
        self,
        user_message: str,
        thread_id: str,
    ) -> dict[str, Any]:
        """向 Agent 发送一轮用户消息。"""

        if not user_message.strip():
            raise ValueError("user_message 不能为空。")

        if not thread_id.strip():
            raise ValueError("thread_id 不能为空。")

        return self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_message.strip(),
                    }
                ]
            },
            config=self._build_config(thread_id),
        )

    def get_history(
        self,
        thread_id: str,
    ) -> list[BaseMessage]:
        """读取某个 thread 的当前压缩后 messages state。"""

        snapshot = self.agent.get_state(
            self._build_config(thread_id)
        )

        if not snapshot.values:
            return []

        return list(
            snapshot.values.get("messages", [])
        )

    def clear_history(self, thread_id: str) -> None:
        """删除某个 thread 的 SQLite 历史。"""

        self.memory_manager.delete_thread(thread_id)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """兼容字符串和部分 provider 的 content blocks。"""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []

            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)

            return "".join(text_parts)

        return str(content)

    @staticmethod
    def get_final_answer(
        result: dict[str, Any],
    ) -> str:
        """提取最后一条不含 tool_calls 的 AIMessage。"""

        messages = result.get("messages", [])

        for message in reversed(messages):
            if (
                isinstance(message, AIMessage)
                and not message.tool_calls
            ):
                return FashionAdvisorAgentService._content_to_text(
                    message.content
                )

        return "未获得有效的最终回答。"

    @staticmethod
    def extract_tool_trace(
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        提取真实 Tool Call / Tool Result。

        这不是模型私有思维链，只是程序实际发生的工具事件，
        非常适合开发调试和运行观测。
        """

        trace: list[dict[str, Any]] = []

        for message in result.get("messages", []):
            if isinstance(message, AIMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
                    trace.append(
                        {
                            "type": "tool_call",
                            "name": tool_call.get("name"),
                            "args": tool_call.get("args", {}),
                            "id": tool_call.get("id"),
                        }
                    )

            elif isinstance(message, ToolMessage):
                trace.append(
                    {
                        "type": "tool_result",
                        "name": getattr(message, "name", None),
                        "content": FashionAdvisorAgentService._content_to_text(
                            message.content
                        ),
                        "tool_call_id": message.tool_call_id,
                    }
                )

        return trace

    def close(self) -> None:
        """程序退出时关闭 SQLite。"""

        self.memory_manager.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

