"""Streamlit Web UI for FashionAdvisor Agent.

Run:
    streamlit run app.py

The UI intentionally exposes only real tool-call traces, not private model reasoning.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Streamlit secrets -> environment variables
# -----------------------------------------------------------------------------
# Local development can use .env. When deployed with Streamlit, the same keys
# can be placed in .streamlit/secrets.toml without changing the application code.
def _load_streamlit_secrets() -> None:
    keys = [
        "MODEL_PROVIDER",
        "MODEL_NAME",
        "MODEL_API_KEY",
        "MODEL_BASE_URL",
        "MODEL_TEMPERATURE",
        "MODEL_MAX_TOKENS",
        "MODEL_MAX_RETRIES",
        "SUMMARY_MODEL_PROVIDER",
        "SUMMARY_MODEL_NAME",
        "SUMMARY_MODEL_API_KEY",
        "SUMMARY_MODEL_BASE_URL",
        "SUMMARY_MODEL_TEMPERATURE",
        "SUMMARY_MODEL_MAX_TOKENS",
        "DASHSCOPE_API_KEY",
        "EMBEDDING_MODEL_NAME",
        "TAVILY_ENABLED",
        "TAVILY_API_KEY",
        "TAVILY_MAX_RESULTS",
        "TAVILY_TOPIC",
        "TAVILY_SEARCH_DEPTH",
    ]

    try:
        secrets = st.secrets
        for key in keys:
            if key not in os.environ and key in secrets:
                os.environ[key] = str(secrets[key])
    except Exception:
        # No secrets.toml is perfectly valid for local .env usage.
        pass


_load_streamlit_secrets()

# Import project modules only after optional secrets have been injected.
from agent.agent_service import FashionAdvisorAgentService  # noqa: E402
from config import settings  # noqa: E402
from rag.knowledge_base import KnowledgeBaseService  # noqa: E402


st.set_page_config(
    page_title="FashionAdvisor Agent",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .block-container {max-width: 1100px; padding-top: 1.7rem;}
        .hero-title {font-size: 2.05rem; font-weight: 760; margin-bottom: 0.15rem;}
        .hero-subtitle {color: #6b7280; margin-bottom: 1rem;}
        .mini-card {
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 12px;
            padding: .85rem 1rem;
            min-height: 95px;
        }
        .mini-card b {font-size: 1rem;}
        .tool-name {font-weight: 700;}
        div[data-testid="stChatMessage"] {border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _knowledge_base_ready() -> bool:
    """A lightweight readiness check that does not call the embedding API."""

    registry = settings.ingested_registry_path
    if not registry.exists():
        return False

    try:
        has_registry = any(
            line.strip()
            for line in registry.read_text(encoding="utf-8").splitlines()
        )
        has_chroma_files = any(
            path.is_file() and path.name != ".gitkeep"
            for path in settings.chroma_directory.rglob("*")
        )
        return has_registry and has_chroma_files
    except OSError:
        return False


def _config_errors() -> list[str]:
    errors: list[str] = []

    if not settings.model_api_key or settings.model_api_key.startswith("请填写"):
        errors.append("缺少 MODEL_API_KEY（主模型）")

    # DashScopeEmbeddings reads DASHSCOPE_API_KEY from the environment.
    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not dashscope_key or dashscope_key.startswith("请填写"):
        errors.append("缺少 DASHSCOPE_API_KEY（Embedding）")

    if settings.tavily_enabled:
        if not settings.tavily_api_key or settings.tavily_api_key.startswith("请填写"):
            errors.append("TAVILY_ENABLED=true，但缺少 TAVILY_API_KEY")

    return errors


@st.cache_resource(show_spinner=False)
def _get_agent_service() -> FashionAdvisorAgentService:
    return FashionAdvisorAgentService(debug=False)


def _format_tool_args(args: Any) -> str:
    if isinstance(args, dict):
        if not args:
            return "{}"
        return "\n".join(f"- **{key}**: `{value}`" for key, value in args.items())
    return str(args)


def _render_tool_trace(trace: list[dict[str, Any]]) -> None:
    if not trace:
        st.caption("本轮由模型直接回答，没有调用外部工具。")
        return

    for index, item in enumerate(trace, start=1):
        if item.get("type") == "tool_call":
            st.markdown(f"**{index}. 调用 `{item.get('name', 'unknown')}`**")
            st.markdown(_format_tool_args(item.get("args", {})))
        elif item.get("type") == "tool_result":
            content = str(item.get("content", ""))
            preview = content if len(content) <= 1800 else content[:1800] + "\n...（已截断）"
            st.code(preview, language="text")


def _render_history(service: FashionAdvisorAgentService, thread_id: str) -> None:
    """Render user messages and final AI answers from the persisted graph state."""

    from langchain_core.messages import AIMessage, HumanMessage

    try:
        messages = service.get_history(thread_id)
    except Exception:
        return

    for message in messages:
        if isinstance(message, HumanMessage):
            content = service._content_to_text(message.content)
            if content:
                with st.chat_message("user"):
                    st.markdown(content)
        elif isinstance(message, AIMessage) and not message.tool_calls:
            content = service._content_to_text(message.content)
            if content:
                with st.chat_message("assistant"):
                    st.markdown(content)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-title">👔 FashionAdvisor Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">基于 LangChain 的智能服装顾问 · Agentic RAG · Local-first · Web fallback · Persistent Memory</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="mini-card"><b>本地知识</b><br>结构感知切分 + Chroma RAG</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="mini-card"><b>检索质量</b><br>Relevance Gate + MMR + Neighbor</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="mini-card"><b>确定性规则</b><br>尺码推荐 + 斤/公斤换算</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="mini-card"><b>外部信息</b><br>Tavily 实时搜索 + SQLite Memory</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("运行控制")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "fashion_user_001"

    thread_id = st.text_input(
        "会话 ID",
        value=st.session_state.thread_id,
        help="同一个 ID 会复用 SQLite 中保存的多轮会话状态。",
    ).strip() or "fashion_user_001"
    st.session_state.thread_id = thread_id

    st.caption(f"模型：`{settings.model_name}`")
    st.caption(f"Embedding：`{settings.embedding_model_name}`")
    st.caption(
        "Tavily：" + ("已启用" if settings.tavily_enabled else "已关闭")
    )

    st.divider()

    kb_ready = _knowledge_base_ready()
    if kb_ready:
        st.success("本地知识库：已初始化")
    else:
        st.warning("本地知识库：尚未初始化")

    if st.button("初始化 / 补充知识库", use_container_width=True):
        errors = _config_errors()
        if errors:
            st.error("无法初始化：\n\n" + "\n".join(f"- {item}" for item in errors))
        else:
            try:
                with st.spinner("正在切分文本、生成 Embedding 并写入 Chroma..."):
                    kb = KnowledgeBaseService()
                    results = []
                    for filename in ("洗涤养护.txt", "颜色选择.txt"):
                        results.append(kb.ingest_file(settings.data_directory / filename))
                st.success("\n".join(results))
                st.rerun()
            except Exception as exc:
                st.error(f"知识库初始化失败：{type(exc).__name__}: {exc}")

    try:
        service_for_clear = _get_agent_service() if not _config_errors() else None
    except Exception:
        service_for_clear = None

    if st.button("清空当前会话", use_container_width=True):
        if service_for_clear is None:
            st.warning("请先完成模型配置。")
        else:
            service_for_clear.clear_history(thread_id)
            st.success("当前会话已清空。")
            st.rerun()

    st.divider()
    st.markdown("**推荐体验问题**")
    st.caption("• 羊毛衫应该怎么洗？")
    st.caption("• 我 178cm、80kg，喜欢宽松，穿什么码？")
    st.caption("• 正式面试穿黑色合适吗？")
    st.caption("• 今年秋冬最新流行色有哪些？")
    st.caption("• 莫代尔面料怎么护理？（可观察本地不足→联网兜底）")


# -----------------------------------------------------------------------------
# Configuration guard
# -----------------------------------------------------------------------------
errors = _config_errors()
if errors:
    st.warning(
        "当前尚未配置完整的模型 / 搜索 API。请复制 `.env.example` 为 `.env` 后填写 Key，"
        "或在 Streamlit `secrets.toml` 中配置同名字段。"
    )
    st.code("\n".join(errors), language="text")
    st.stop()

if not _knowledge_base_ready():
    st.info(
        "首次运行请先在左侧点击 **初始化 / 补充知识库**，"
        "或在终端执行 `python scripts/init_knowledge_base.py`。"
    )


# -----------------------------------------------------------------------------
# Chat UI
# -----------------------------------------------------------------------------
try:
    agent_service = _get_agent_service()
except Exception as exc:
    st.error(f"Agent 初始化失败：{type(exc).__name__}: {exc}")
    st.stop()

_render_history(agent_service, thread_id)

sample_columns = st.columns(4)
sample_prompts = [
    "羊毛衫应该怎么洗？",
    "我178cm、80kg，喜欢宽松，穿什么码？",
    "正式面试穿黑色合适吗？",
    "今年秋冬最新流行色有哪些？",
]
selected_prompt: str | None = None
for column, sample in zip(sample_columns, sample_prompts):
    with column:
        if st.button(sample, use_container_width=True):
            selected_prompt = sample

user_input = st.chat_input("输入你的服装问题，例如：我178cm、80kg，喜欢宽松，面试穿黑色合适吗？")
prompt = selected_prompt or user_input

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Agent 正在判断需要使用哪些能力..."):
                result = agent_service.invoke(
                    user_message=prompt,
                    thread_id=thread_id,
                )
                final_answer = agent_service.get_final_answer(result)
                trace = agent_service.extract_tool_trace(result)

            st.markdown(final_answer)

            with st.expander("查看本轮 Tool Trace", expanded=False):
                st.caption("这里只展示实际工具调用与工具返回，不展示模型私有思维链。")
                _render_tool_trace(trace)

    except Exception as exc:
        st.error(f"本轮请求失败：{type(exc).__name__}: {exc}")
        st.caption("请检查 API Key、网络状态、知识库是否已初始化，以及 `.env` 参数配置。")
