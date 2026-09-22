"""
Agent 端到端测试。

运行前：
1. 配好 .env；
2. python scripts/init_knowledge_base.py
3. python tests/test_agent.py

重点观察 Tool Trace，而不仅仅是最终答案。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uuid import uuid4

from agent.agent_service import FashionAdvisorAgentService


TEST_CASES = [
    # 普通聊天：不应调用工具
    "你好",

    # 确定性工具
    "160斤是多少公斤？",
    "我178cm、160斤，应该穿什么尺码？",

    # 本地 RAG
    "羊毛/羊绒衣物应该怎么洗和养护？",

    # 多 Tool：尺码 + 本地知识
    "我178cm、80kg，喜欢宽松一点，面试穿黑色合适吗？",

    # 明显时效性：应该优先考虑 web_search
    "帮我查一下2026年秋冬最新的服装流行色趋势，并告诉我来源。",

    # 本地库可能不覆盖：理想路径为 local RAG -> insufficient -> web_search
    "莫代尔面料日常应该如何洗涤和保养？如果本地知识不够就联网查。",
]


def main() -> None:
    with FashionAdvisorAgentService(debug=False) as service:
        for index, question in enumerate(TEST_CASES, start=1):
            thread_id = f"agent_test_{index}_{uuid4().hex}"

            print("=" * 90)
            print(f"问题 {index}：{question}")

            result = service.invoke(
                user_message=question,
                thread_id=thread_id,
            )

            print("\nTool Trace:")
            trace = service.extract_tool_trace(result)

            if not trace:
                print("本轮没有调用 Tool。")
            else:
                for item in trace:
                    print(item)

            print("\nFinal Answer:")
            print(service.get_final_answer(result))


if __name__ == "__main__":
    main()
