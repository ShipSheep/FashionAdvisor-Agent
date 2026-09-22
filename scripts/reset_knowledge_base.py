"""
重置当前 当前 Chroma collection。

当你修改 CHUNK_SIZE / CHUNK_OVERLAP / Chunk 算法后，
建议运行：
    python scripts/reset_knowledge_base.py
    python scripts/init_knowledge_base.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.knowledge_base import KnowledgeBaseService


def main() -> None:
    service = KnowledgeBaseService()
    service.reset()
    print("知识库 collection 和入库登记已经重置。")


if __name__ == "__main__":
    main()
