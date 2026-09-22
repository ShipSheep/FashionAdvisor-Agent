"""
结构感知的 TXT Chunk 切分器。

为什么不直接“每 800 个字符切一次”？
---------------------------------
固定长度硬切会出现：
    ...羊毛衫洗涤方式前半句 | 后半句和养护规则...

这会破坏一个完整知识点。

本项目采用四层保护：
1. 先识别文档天然结构：
   - “一、春季服装”这种一级标题；
   - “1. 纯棉材质”这种编号条目。
2. 一个逻辑块如果本来就小于 chunk_size，整个保留，不再硬切。
3. 只有逻辑块过长时，才用 RecursiveCharacterTextSplitter，
   并优先按照：自然段 -> 换行 -> 中文句号/问号/分号 -> 逗号 -> 字符切。
4. 对超长逻辑块保留 chunk_overlap。

另外，检索阶段还会在 retriever.py 中把命中 chunk 的前后邻居补回来，
进一步降低 chunk 边界造成的上下文损失。
"""

from dataclasses import dataclass
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


# 匹配：一、xxx / 二、xxx / 十、xxx
_CHINESE_HEADING = re.compile(
    r"^\s*[一二三四五六七八九十百]+[、．.]\s*.+$"
)

# 匹配：1. xxx / 2、xxx / 3．xxx
_NUMBERED_HEADING = re.compile(
    r"^\s*\d+\s*[、．.]\s*.+$"
)


@dataclass
class LogicalSection:
    """一块尽量保持语义完整的文档逻辑单元。"""

    parent_title: str
    title: str
    body: str


class StructureAwareTextSplitter:
    """“结构优先 + 递归兜底”的中文 TXT 切分器。"""

    def __init__(self) -> None:
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

        # 中文场景下，优先保留较大的自然语言单位。
        # keep_separator="end" 尽量把标点留在前一句结尾，阅读更自然。
        self.separators = [
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            ".",
            "!",
            "?",
            ";",
            "，",
            "、",
            ",",
            " ",
            "",
        ]

    @staticmethod
    def _normalize_text(text: str) -> str:
        """统一换行符并去掉文档首尾多余空白。"""
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    def _extract_logical_sections(self, text: str) -> list[LogicalSection]:
        """
        先按照显式标题/编号识别逻辑结构。

        例如洗涤知识：
            一、春季服装
            1. 纯棉材质
            洗涤：...
            养护：...

        会被识别成一个 LogicalSection：
            parent_title = 一、春季服装
            title        = 1. 纯棉材质
            body         = 洗涤... + 养护...

        这样“纯棉的洗涤 + 养护”天然留在一起。
        """

        lines = self._normalize_text(text).split("\n")

        sections: list[LogicalSection] = []
        parent_title = ""
        current_title = ""
        current_lines: list[str] = []

        def flush_current() -> None:
            """把当前累积内容保存成一个逻辑块。"""
            nonlocal current_lines

            body = "\n".join(current_lines).strip()

            # 如果只有一个一级标题、下面马上就是二级条目，
            # 不把“裸标题”单独做成 chunk。
            if not body:
                current_lines = []
                return

            title = current_title or parent_title or "正文"
            sections.append(
                LogicalSection(
                    parent_title=parent_title,
                    title=title,
                    body=body,
                )
            )
            current_lines = []

        for raw_line in lines:
            stripped = raw_line.strip()

            # 保留空行，它对自然段识别有帮助。
            if not stripped:
                current_lines.append("")
                continue

            # 一级中文标题：更新 parent_title。
            if _CHINESE_HEADING.match(stripped):
                flush_current()
                parent_title = stripped
                current_title = stripped
                continue

            # 数字编号标题：在当前一级标题下开启新条目。
            if _NUMBERED_HEADING.match(stripped):
                flush_current()
                current_title = stripped
                continue

            current_lines.append(raw_line.rstrip())

        flush_current()

        # 如果文档本身没有标题结构，就退化成一个大逻辑块，
        # 后续再交给 RecursiveCharacterTextSplitter。
        if not sections:
            normalized = self._normalize_text(text)
            if normalized:
                sections.append(
                    LogicalSection(
                        parent_title="",
                        title="正文",
                        body=normalized,
                    )
                )

        return sections

    @staticmethod
    def _build_prefix(section: LogicalSection) -> str:
        """
        给每个 chunk 保留所属标题。

        即使一个超长 section 被拆成多个 child chunks，
        每个 child chunk 都会重复带上标题，例如：
            三、秋季服装
            1. 羊毛/羊绒材质

        这会显著降低“正文被切出来后不知道自己属于什么主题”的问题。
        """

        titles: list[str] = []

        if section.parent_title:
            titles.append(section.parent_title)

        if section.title and section.title != section.parent_title:
            titles.append(section.title)

        return "\n".join(titles).strip()

    def _split_one_section(
        self,
        section: LogicalSection,
    ) -> list[str]:
        """对单个逻辑块执行必要的递归切分。"""

        prefix = self._build_prefix(section)
        body = section.body.strip()

        full_text = f"{prefix}\n{body}".strip() if prefix else body

        # 关键点：小于 chunk_size 时完全不切。
        # 因此 chunk 并不是固定 800 字。
        if len(full_text) <= self.chunk_size:
            return [full_text]

        # 超长逻辑块才进入递归切分。
        # 给标题预留空间，避免最终 prefix + part 大幅超过 chunk_size。
        prefix_cost = len(prefix) + 1 if prefix else 0
        body_chunk_size = max(200, self.chunk_size - prefix_cost)

        # overlap 不宜超过 child chunk 的 1/3。
        safe_overlap = min(
            self.chunk_overlap,
            max(0, body_chunk_size // 3),
        )

        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_chunk_size,
            chunk_overlap=safe_overlap,
            separators=self.separators,
            keep_separator="end",
            length_function=len,
        )

        body_parts = fallback_splitter.split_text(body)

        result: list[str] = []
        for part in body_parts:
            chunk_text = f"{prefix}\n{part}".strip() if prefix else part.strip()
            if chunk_text:
                result.append(chunk_text)

        return result

    def split_document(self, document: Document) -> list[Document]:
        """
        把一整份 Document 切成最终可以写入 Chroma 的 chunks。

        metadata 会额外记录：
        - section_title
        - parent_title
        - section_index
        - section_chunk_index
        - section_total_chunks

        全局 chunk_index / chunk_id 会在 KnowledgeBaseService 中统一补充。
        """

        logical_sections = self._extract_logical_sections(
            document.page_content
        )

        output: list[Document] = []

        for section_index, section in enumerate(logical_sections):
            child_texts = self._split_one_section(section)

            for section_chunk_index, child_text in enumerate(child_texts):
                metadata = dict(document.metadata)
                metadata.update(
                    {
                        "parent_title": section.parent_title,
                        "section_title": section.title,
                        "section_index": section_index,
                        "section_chunk_index": section_chunk_index,
                        "section_total_chunks": len(child_texts),
                    }
                )

                output.append(
                    Document(
                        page_content=child_text,
                        metadata=metadata,
                    )
                )

        return output
