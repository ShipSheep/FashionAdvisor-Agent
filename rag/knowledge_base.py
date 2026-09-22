"""
知识库入库服务。

完整流程：
原始 TXT
    -> 内容标准化
    -> 生成与“文本 + Chunk配置”绑定的 document_id
    -> StructureAwareTextSplitter
    -> 为 chunks 添加确定性 chunk_id / metadata
    -> Embedding
    -> Chroma

特别说明：
项目原始数据包含 3 个 TXT：
1. 洗涤养护.txt       -> 适合非结构化 RAG
2. 颜色选择.txt       -> 适合非结构化 RAG
3. 尺码推荐.txt       -> 本项目故意不作为主要 RAG 数据，而转成 size_chart.json
                         由 recommend_size Tool 确定性计算。

原因是尺码表属于结构化规则，让 LLM 从向量文本中“猜”尺码反而不可靠。
"""

from datetime import datetime
import hashlib
from pathlib import Path

from langchain_core.documents import Document

from config import settings
from rag.text_splitter import StructureAwareTextSplitter
from rag.vector_store import VectorStoreService


class KnowledgeBaseService:
    """负责文档切分、去重和入库。"""

    def __init__(self) -> None:
        settings.data_directory.mkdir(parents=True, exist_ok=True)
        settings.chroma_directory.mkdir(parents=True, exist_ok=True)

        self.vector_store_service = VectorStoreService()
        self.text_splitter = StructureAwareTextSplitter()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    @staticmethod
    def _build_document_id(text: str, source: str) -> str:
        """
        document_id 不只依赖原始文本，还依赖 Chunk 策略。

        为什么？
        如果你把 CHUNK_SIZE 从 800 改成 500，虽然原文本没变，
        但实际上已经应该重新切分和重新生成向量。
        """

        fingerprint_source = "|".join(
            [
                settings.chunking_version,
                str(settings.chunk_size),
                str(settings.chunk_overlap),
                source,
                text,
            ]
        )

        return hashlib.md5(
            fingerprint_source.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _is_ingested(document_id: str) -> bool:
        path = settings.ingested_registry_path

        if not path.exists():
            return False

        ids = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

        return document_id in ids

    @staticmethod
    def _mark_ingested(document_id: str) -> None:
        settings.ingested_registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with settings.ingested_registry_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(document_id + "\n")

    def build_chunks(
        self,
        text: str,
        source: str,
    ) -> list[Document]:
        """
        只执行“切分”，不写入 Chroma。

        scripts/preview_chunks.py 会直接调用它，
        因此你可以在真正入库前亲眼看每个 TXT 被切成什么样。
        """

        normalized = self._normalize_text(text)

        if not normalized:
            return []

        document_id = self._build_document_id(
            text=normalized,
            source=source,
        )

        original_document = Document(
            page_content=normalized,
            metadata={
                "source": source,
                "document_id": document_id,
                "chunking_version": settings.chunking_version,
                "create_time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            },
        )

        chunks = self.text_splitter.split_document(
            original_document
        )

        total_chunks = len(chunks)

        for chunk_index, chunk in enumerate(chunks):
            # 确定性 ID：同一 document 的 chunk 具有连续编号。
            # 后续 Retriever 才能很容易获取 idx-1 / idx+1。
            chunk_id = f"{document_id}_{chunk_index:04d}"

            chunk.metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                }
            )

        return chunks

    def ingest_text(self, text: str, source: str) -> str:
        """将一整份 TXT 内容切分并写入 Chroma。"""

        normalized = self._normalize_text(text)

        if not normalized:
            return f"[跳过] {source} 内容为空。"

        document_id = self._build_document_id(
            text=normalized,
            source=source,
        )

        if self._is_ingested(document_id):
            return f"[跳过] {source} 当前版本已经导入过。"

        chunks = self.build_chunks(
            text=normalized,
            source=source,
        )

        ids = [
            str(chunk.metadata["chunk_id"])
            for chunk in chunks
        ]

        self.vector_store_service.add_documents(
            documents=chunks,
            ids=ids,
        )

        self._mark_ingested(document_id)

        return (
            f"[成功] {source} 已导入，"
            f"共生成 {len(chunks)} 个结构化文本块。"
        )

    def ingest_file(self, file_path: str | Path) -> str:
        """导入 UTF-8 TXT 文件。"""

        path = Path(file_path)
        text = path.read_text(encoding="utf-8")

        return self.ingest_text(
            text=text,
            source=path.name,
        )

    def reset(self) -> None:
        """清空向量 collection 和本地入库登记。"""

        self.vector_store_service.reset_collection()

        if settings.ingested_registry_path.exists():
            settings.ingested_registry_path.unlink()
