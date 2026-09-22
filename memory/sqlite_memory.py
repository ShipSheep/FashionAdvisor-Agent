"""
LangGraph SQLite Checkpointer。

SQLite 的职责：持久化 Agent state。
其中不仅包括 HumanMessage / AIMessage，还可能包括：
- AIMessage 中的 tool_calls；
- ToolMessage；
- SummarizationMiddleware 写回的历史摘要；
- LangGraph checkpoint 信息。

因此不需要再维护第二套 FileChatMessageHistory / SQLChatMessageHistory。
"""

import sqlite3

from config import settings
from langgraph.checkpoint.sqlite import SqliteSaver


class SQLiteMemoryManager:
    """负责 SQLite 连接与 SqliteSaver 生命周期。"""

    def __init__(self) -> None:
        settings.sqlite_db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # LangGraph SqliteSaver 内部使用锁。
        # check_same_thread=False 也方便未来换成 Web/Streamlit 调用。
        self.connection = sqlite3.connect(
            str(settings.sqlite_db_path),
            check_same_thread=False,
        )

        self.checkpointer = SqliteSaver(
            self.connection
        )

    def delete_thread(self, thread_id: str) -> None:
        """删除某个会话的全部 checkpoints。"""
        self.checkpointer.delete_thread(thread_id)

    def close(self) -> None:
        """关闭 SQLite 连接。"""
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
