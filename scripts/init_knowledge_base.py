"""
初始化正式 RAG 知识库。

运行：
    python scripts/init_knowledge_base.py

正式入库：
    - 洗涤养护.txt
    - 颜色选择.txt

不入库：
    - 尺码推荐_原始数据.txt
      因为它已经结构化为 size_chart.json + recommend_size Tool。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from rag.knowledge_base import KnowledgeBaseService


RAG_FILES = [
    settings.data_directory / "洗涤养护.txt",
    settings.data_directory / "颜色选择.txt",
]


def main() -> None:
    service = KnowledgeBaseService()

    print("开始初始化 RAG 知识库...\n")

    for file_path in RAG_FILES:
        if not file_path.exists():
            print(f"[缺失] {file_path}")
            continue

        result = service.ingest_file(file_path)
        print(result)

    print("\n知识库初始化完成。")


if __name__ == "__main__":
    main()
