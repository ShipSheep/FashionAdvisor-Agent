"""
知识检索服务。

本地 RAG 流程：

Query
  -> Relevance Gate
       |- 不足：返回 LOCAL_KB_STATUS=INSUFFICIENT
       `- 足够：继续
  -> MMR 找 anchor chunks
  -> Neighbor Expansion
  -> 返回本地证据

为什么一定要加 Relevance Gate？
--------------------------------
向量数据库面对一个完全不相关的问题，也往往会从已有文档中“硬找”几个最像的结果。
因此：
    documents != []
并不等于：
    documents 真的有用。

Gate 就是专门解决这个问题的。
"""

from dataclasses import dataclass

from langchain_core.documents import Document

from config import settings
from rag.vector_store import VectorStoreService


@dataclass
class RetrievalResult:
    """Retriever 返回给 Tool 的结构化结果。"""

    documents: list[Document]
    sufficient: bool
    best_score: float | None
    reason: str


class KnowledgeRetriever:
    """向 Agent Tool 提供“有相关性判断”的本地知识检索。"""

    def __init__(self) -> None:
        self.vector_store_service = VectorStoreService()

    @staticmethod
    def _neighbor_ids(anchor: Document) -> list[str]:
        """根据 anchor metadata 计算前后邻居的确定性 chunk ID。"""

        document_id = str(anchor.metadata.get("document_id", ""))
        chunk_index = int(anchor.metadata.get("chunk_index", 0))
        total_chunks = int(anchor.metadata.get("total_chunks", 1))

        if not document_id:
            return []

        start = max(
            0,
            chunk_index - settings.retrieval_neighbor_window,
        )
        end = min(
            total_chunks - 1,
            chunk_index + settings.retrieval_neighbor_window,
        )

        return [
            f"{document_id}_{index:04d}"
            for index in range(start, end + 1)
        ]

    def inspect_relevance(
        self,
        query: str,
    ) -> list[tuple[Document, float]]:
        """
        暴露 Gate 的原始评分，方便调参与调试观察。

        正式 Agent 不需要直接调用这个方法，
        scripts/inspect_relevance.py 会使用它。
        """

        return self.vector_store_service.relevance_search(query)

    def search(self, query: str) -> RetrievalResult:
        """
        先判断本地知识库是否有“足够相关”的内容，再构建上下文。
        """

        # ----------------------------------------------------------
        # Step 1. Relevance Gate
        # ----------------------------------------------------------
        scored_docs = self.vector_store_service.relevance_search(query)

        if not scored_docs:
            return RetrievalResult(
                documents=[],
                sufficient=False,
                best_score=None,
                reason="本地知识库为空，或当前查询没有获得任何候选文档。",
            )

        # similarity_search_with_relevance_scores 的语义是：
        # 通常 1 更相关、0 更不相关，因此取最大值。
        best_score = max(score for _, score in scored_docs)

        if best_score < settings.retrieval_relevance_threshold:
            return RetrievalResult(
                documents=[],
                sufficient=False,
                best_score=best_score,
                reason=(
                    "本地知识库虽然返回了最相近文档，但最高相关度低于阈值，"
                    "因此不把这些文档当作可靠证据。"
                ),
            )

        # ----------------------------------------------------------
        # Step 2. MMR Anchor Retrieval
        # ----------------------------------------------------------
        anchors = self.vector_store_service.mmr_search(query)

        if not anchors:
            return RetrievalResult(
                documents=[],
                sufficient=False,
                best_score=best_score,
                reason="相关性 Gate 通过，但 MMR 没有返回可用 anchor。",
            )

        # ----------------------------------------------------------
        # Step 3. Neighbor Expansion
        # ----------------------------------------------------------
        result: list[Document] = []
        seen_ids: set[str] = set()

        for anchor_rank, anchor in enumerate(anchors, start=1):
            neighbor_ids = self._neighbor_ids(anchor)
            neighbors = self.vector_store_service.get_by_ids(neighbor_ids)

            # get_by_ids 不保证顺序，重新按原文 chunk_index 排序。
            neighbors.sort(
                key=lambda doc: int(
                    doc.metadata.get("chunk_index", 0)
                )
            )

            anchor_id = str(anchor.metadata.get("chunk_id", ""))

            for document in neighbors:
                chunk_id = str(document.metadata.get("chunk_id", ""))

                if chunk_id in seen_ids:
                    continue

                metadata = dict(document.metadata)
                metadata["anchor_rank"] = anchor_rank
                metadata["retrieval_role"] = (
                    "anchor" if chunk_id == anchor_id else "neighbor"
                )
                metadata["local_best_relevance_score"] = round(
                    best_score,
                    4,
                )

                result.append(
                    Document(
                        page_content=document.page_content,
                        metadata=metadata,
                    )
                )
                seen_ids.add(chunk_id)

        return RetrievalResult(
            documents=result,
            sufficient=True,
            best_score=best_score,
            reason="本地知识库相关性 Gate 通过。",
        )

    @staticmethod
    def format_result(result: RetrievalResult) -> str:
        """
        格式化 Tool 返回内容。

        特意加入机器可读状态标记：
            LOCAL_KB_STATUS=SUFFICIENT
        或：
            LOCAL_KB_STATUS=INSUFFICIENT

        Agent Prompt 会根据这个标记决定是否继续调用 web_search。
        """

        score_text = (
            "N/A"
            if result.best_score is None
            else f"{result.best_score:.4f}"
        )

        if not result.sufficient:
            return "\n".join(
                [
                    "LOCAL_KB_STATUS=INSUFFICIENT",
                    f"LOCAL_KB_BEST_SCORE={score_text}",
                    f"LOCAL_KB_THRESHOLD={settings.retrieval_relevance_threshold:.4f}",
                    f"原因：{result.reason}",
                    "建议：如果用户询问的是公开信息，可考虑调用 web_search；"
                    "如果属于本系统内部规则或私有知识，不要用公网内容冒充内部依据。",
                ]
            )

        blocks: list[str] = [
            "LOCAL_KB_STATUS=SUFFICIENT",
            f"LOCAL_KB_BEST_SCORE={score_text}",
            f"LOCAL_KB_THRESHOLD={settings.retrieval_relevance_threshold:.4f}",
            f"说明：{result.reason}",
        ]

        for index, document in enumerate(result.documents, start=1):
            metadata = document.metadata

            source = metadata.get("source", "未知来源")
            parent_title = metadata.get("parent_title", "")
            section_title = metadata.get("section_title", "")
            chunk_index = metadata.get("chunk_index", "?")
            role = metadata.get("retrieval_role", "unknown")

            blocks.append(
                "\n".join(
                    [
                        f"[本地资料 {index}]",
                        f"来源：{source}",
                        f"一级主题：{parent_title or '-'}",
                        f"知识条目：{section_title or '-'}",
                        f"文本块：{chunk_index}",
                        f"检索角色：{role}",
                        "内容：",
                        document.page_content,
                    ]
                )
            )

        return "\n\n".join(blocks)
