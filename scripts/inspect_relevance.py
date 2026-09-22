"""
观察本地知识库的 relevance score，帮助校准阈值。

为什么需要这个脚本？
--------------------
RETRIEVAL_RELEVANCE_THRESHOLD=0.45 只是一个起始值。
不同 Embedding 模型、文档规模、文档语言都会影响分数分布。

正确做法：
1. 准备几条“应该命中本地库”的问题；
2. 准备几条“明显不在本地库”的问题；
3. 看两组分数是否有明显间隔；
4. 把阈值设置在两组分布之间。

运行前：
    python scripts/init_knowledge_base.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from rag.retriever import KnowledgeRetriever


QUERIES = [
    # 应该命中本地知识库
    "羊毛和羊绒衣物应该怎么洗？",
    "正式面试穿黑色合适吗？",
    "黄皮肤适合什么服装颜色？",

    # 明显不属于当前本地知识库
    "2026年秋冬最新国际流行色有哪些？",
    "MacBook Pro 应该怎么充电？",
    "东京明天天气怎么样？",
]


def main() -> None:
    retriever = KnowledgeRetriever()

    print("当前 Gate 阈值：", settings.retrieval_relevance_threshold)
    print()

    for query in QUERIES:
        print("=" * 90)
        print("Query:", query)

        scored_docs = retriever.inspect_relevance(query)

        if not scored_docs:
            print("没有候选结果。")
            continue

        for rank, (doc, score) in enumerate(scored_docs, start=1):
            print(
                f"#{rank} score={score:.4f} "
                f"source={doc.metadata.get('source')} "
                f"section={doc.metadata.get('section_title')}"
            )

        result = retriever.search(query)
        print(
            f"Gate 结果：sufficient={result.sufficient}, "
            f"best_score={result.best_score}"
        )
        print("原因：", result.reason)


if __name__ == "__main__":
    main()
