"""
Chroma 向量数据库访问层。

该层保持职责单一：
- 写入 chunks；
- MMR 检索；
- Relevance Score 检索；
- 按确定性 ID 获取邻接 chunks；
- 重置 collection。

注意：
    “是否该联网搜索”不在这里决策。
这里只负责把可靠的检索信号提供给上层 Retriever / Agent。
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from models.model_factory import create_embedding_model


class VectorStoreService:
    """封装 Chroma 的基础操作。"""

    def __init__(self) -> None:
        settings.chroma_directory.mkdir(parents=True, exist_ok=True)

        self.embedding_model = create_embedding_model()

        self.vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=self.embedding_model,
            persist_directory=str(settings.chroma_directory),
        )

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        """将 chunks 写入 Chroma。"""

        return self.vector_store.add_documents(
            documents=documents,
            ids=ids,
        )

    def relevance_search(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        返回 Document + 归一化 relevance score。

        这个方法用于“相关性 Gate”，不是最终上下文构建。

        score 越接近 1，通常表示越相关；越接近 0 越不相关。
        需要强调：阈值和 Embedding 模型、语料规模都有关，因此应通过
        scripts/inspect_relevance.py 在自己的数据上做校准。
        """

        gate_k = k if k is not None else settings.retrieval_gate_k

        return self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=gate_k,
        )

    def mmr_search(
        self,
        query: str,
        k: int | None = None,
    ) -> list[Document]:
        """
        使用 MMR 搜索最终 anchor chunks。

        Gate 负责回答“本地知识够不够相关”；
        MMR 负责回答“如果够相关，最终选哪些互补的 chunks”。
        """

        top_k = k if k is not None else settings.retrieval_top_k

        return self.vector_store.max_marginal_relevance_search(
            query=query,
            k=top_k,
            fetch_k=settings.retrieval_fetch_k,
            lambda_mult=settings.retrieval_mmr_lambda,
        )

    def get_by_ids(self, ids: list[str]) -> list[Document]:
        """根据 Chroma document id 精确读取 chunks。"""

        if not ids:
            return []

        return self.vector_store.get_by_ids(ids)

    def reset_collection(self) -> None:
        """清空当前 collection。Chunk 策略改变后可重新构建。"""

        self.vector_store.reset_collection()
