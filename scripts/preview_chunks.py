"""
在真正写入 Chroma 之前，预览 TXT 会如何被切分。

运行：
    python scripts/preview_chunks.py

这个脚本不调用 Chat Model，也不会写入 Chroma。
它的目的就是帮助你直观看懂 Chunk 算法。
"""

from pathlib import Path
import sys

# 允许从项目根目录执行：python scripts/preview_chunks.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from config import settings
from rag.text_splitter import StructureAwareTextSplitter


FILES = [
    settings.data_directory / "洗涤养护.txt",
    settings.data_directory / "颜色选择.txt",
    settings.data_directory / "尺码推荐_原始数据.txt",
]


def main() -> None:
    splitter = StructureAwareTextSplitter()

    print("=" * 80)
    print("Chunk Preview")
    print(
        f"chunk_size={settings.chunk_size}, "
        f"chunk_overlap={settings.chunk_overlap}"
    )
    print("=" * 80)

    for file_path in FILES:
        text = file_path.read_text(encoding="utf-8")

        document = Document(
            page_content=text,
            metadata={
                "source": file_path.name,
            },
        )

        chunks = splitter.split_document(document)

        print(f"\n\n{'#' * 80}")
        print(f"文件：{file_path.name}")
        print(f"共切成：{len(chunks)} 个 chunk")
        print(f"{'#' * 80}")

        for index, chunk in enumerate(chunks):
            print(f"\n--- Chunk {index} / 长度 {len(chunk.page_content)} ---")
            print(
                f"parent_title = {chunk.metadata.get('parent_title')}"
            )
            print(
                f"section_title = {chunk.metadata.get('section_title')}"
            )
            print(chunk.page_content)

    print("\n说明：尺码 TXT 这里只做切分演示，正式项目不会把它作为主要 RAG 数据入库。")


if __name__ == "__main__":
    main()
