"""
本地 RAG Relevance Gate 的人工观察测试。

这不是为了断言一个固定 score，因为不同 embedding 版本的分数可能变化。
重点检查：
- 本地相关问题应更容易 sufficient=True；
- 明显无关问题应更容易 sufficient=False。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import KnowledgeRetriever


TEST_CASES = [
    "羊毛衣物怎么洗？",
    "正式场合应该选择什么颜色？",
    "MacBook Pro 怎么充电？",
]


def main() -> None:
    retriever = KnowledgeRetriever()

    for query in TEST_CASES:
        print("=" * 80)
        print("问题：", query)

        result = retriever.search(query)
        print("sufficient:", result.sufficient)
        print("best_score:", result.best_score)
        print("reason:", result.reason)
        print()
        print(retriever.format_result(result)[:1200])


if __name__ == "__main__":
    main()
