"""不需要 API 的 Chunk 单元测试。"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from rag.text_splitter import StructureAwareTextSplitter


def test_structure_aware_splitter() -> None:
    text = """
一、秋季服装

1. 羊毛/羊绒材质

洗涤：优先干洗；手洗使用羊毛专用洗涤剂。

养护：平铺阴干，避免悬挂拉伸。

2. 厚牛仔材质

洗涤：翻面清洗，减少褪色。

养护：翻面阴干。
""".strip()

    splitter = StructureAwareTextSplitter()

    chunks = splitter.split_document(
        Document(
            page_content=text,
            metadata={"source": "test.txt"},
        )
    )

    assert len(chunks) == 2

    assert "羊毛/羊绒材质" in chunks[0].page_content
    assert "洗涤：" in chunks[0].page_content
    assert "养护：" in chunks[0].page_content

    assert "厚牛仔材质" in chunks[1].page_content

    print("test_structure_aware_splitter PASSED")


if __name__ == "__main__":
    test_structure_aware_splitter()
