"""
命令行聊天入口。

该文件提供命令行入口；Web 交互入口见 app.py。

运行：
    python main.py

支持命令：
    :history  查看当前 SQLite state 中的消息
    :clear    删除当前 thread 的全部历史
    :quit     退出程序
"""

from agent.agent_service import FashionAdvisorAgentService


def print_tool_trace(trace: list[dict]) -> None:
    """打印真实 Tool Call / Tool Result。"""

    if not trace:
        print("[Agent] 本轮没有调用工具。")
        return

    print("\n[Agent Tool Trace]")

    for item in trace:
        if item["type"] == "tool_call":
            print(f"  -> 调用工具: {item['name']}")
            print(f"     参数: {item['args']}")

        elif item["type"] == "tool_result":
            print(f"  <- 工具返回 ({item.get('name')}):")

            content = item["content"]
            preview = (
                content
                if len(content) <= 700
                else content[:700] + "..."
            )
            print(f"     {preview}")


def print_history(
    agent_service: FashionAdvisorAgentService,
    thread_id: str,
) -> None:
    """
    查看当前 thread 的 messages。

    如果摘要已经触发，你会看到历史不再无限增长，
    而是出现“摘要后的消息 + 最近消息”。
    """

    messages = agent_service.get_history(thread_id)

    print(
        f"\n当前会话 state 中共有 {len(messages)} 条 Message："
    )

    for index, message in enumerate(messages, start=1):
        role = message.__class__.__name__
        content = agent_service._content_to_text(
            message.content
        )

        print(f"\n[{index}] {role}")
        print(content)

    print()


def main() -> None:
    print("=" * 72)
    print("FashionAdvisor Agent - CLI")
    print("RAG: 结构感知 Chunk + Relevance Gate + MMR + 邻居扩展")
    print("Search: Local RAG + Tavily Web Fallback\nMemory: SQLite + 自动历史摘要")
    print("=" * 72)

    thread_id = input(
        "请输入会话 ID（直接回车默认 user_001）："
    ).strip()
    thread_id = thread_id or "user_001"

    print(f"当前 thread_id = {thread_id}")
    print(
        "命令：:history 查看记忆，:clear 清空会话，:quit 退出。\n"
    )

    agent_service = FashionAdvisorAgentService(debug=False)

    try:
        while True:
            user_input = input("你：").strip()

            if not user_input:
                continue

            if user_input == ":quit":
                break

            if user_input == ":history":
                print_history(
                    agent_service,
                    thread_id,
                )
                continue

            if user_input == ":clear":
                agent_service.clear_history(thread_id)
                print("当前会话已经从 SQLite 删除。\n")
                continue

            try:
                result = agent_service.invoke(
                    user_message=user_input,
                    thread_id=thread_id,
                )

                print_tool_trace(
                    agent_service.extract_tool_trace(
                        result
                    )
                )

                final_answer = (
                    agent_service.get_final_answer(
                        result
                    )
                )

                print(f"\n助手：{final_answer}\n")

            except Exception as exc:
                print(
                    f"\n[ERROR] {type(exc).__name__}: {exc}\n"
                )

    finally:
        agent_service.close()
        print("程序已退出，SQLite 连接已关闭。")


if __name__ == "__main__":
    main()
