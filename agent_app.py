import re
from io import BytesIO
import json
from typing import List
from datetime import datetime

import pandas as pd
import streamlit as st
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader
import docx


# ---------------------------------------------------------
# Data models
# ---------------------------------------------------------
class FeatureItem(BaseModel):
    module: str = Field(description="所属模块或业务域")
    feature: str = Field(description="功能名称")
    description: str = Field(default="", description="需求摘要")
    acceptance: List[str] = Field(default_factory=list, description="关键验收点或规则")
    dependencies: List[str] = Field(default_factory=list, description="依赖/前提")


class FeatureCollection(BaseModel):
    features: List[FeatureItem]


class TestCase(BaseModel):
    case_id: str
    module: str
    feature: str
    title: str
    precondition: str
    steps: str
    expected: str
    priority: str
    type: str


class TestSuite(BaseModel):
    cases: List[TestCase]


# ---------------------------------------------------------
# File helpers
# ---------------------------------------------------------
def extract_text(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    text = ""

    if ext == "pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    elif ext == "docx":
        document = docx.Document(uploaded_file)
        for para in document.paragraphs:
            text += para.text + "\n"
    elif ext in {"txt", "md"}:
        text = uploaded_file.read().decode("utf-8")
    else:
        raise ValueError("仅支持 PDF / DOCX / TXT / MD 文件")

    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not cleaned:
        raise ValueError("未能从文档中提取文本，请确认文件内容。")
    return cleaned


def chunk_text(text, chunk_size=1800, overlap=200):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


# ---------------------------------------------------------
# Agent steps
# ---------------------------------------------------------
def init_llm(api_key, base_url, model_name, temperature):
    if not api_key:
        st.error("请先在侧边栏配置 API Key")
        return None
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url or None,
        model=model_name,
        temperature=temperature,
        max_retries=2,
    )


def _normalize_feature_response(result):
    """
    将不同形态的模型返回值统一转换为 FeatureItem 列表。
    """
    if result is None:
        return []

    raw_items = []
    if hasattr(result, "features"):
        raw_items = getattr(result, "features", [])
    elif isinstance(result, dict):
        raw_items = result.get("features", [])
    elif isinstance(result, list):
        raw_items = result
    else:
        raw_items = []

    normalized = []
    for item in raw_items:
        if isinstance(item, FeatureItem):
            normalized.append(item)
        elif isinstance(item, dict):
            try:
                normalized.append(FeatureItem(**item))
            except ValidationError:
                continue
    return normalized


def analyze_features(llm, text, debug: bool = False, log_fn=None):
    parser = JsonOutputParser(pydantic_object=FeatureCollection)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是资深需求分析师，请从需求片段中提取功能模块与功能项。"
                   "要求聚焦业务目标，补充关键验收点/校验规则。"
                   "\n{format_instructions}"),
        ("human", "需求片段（ID: {segment_id}）：\n{segment_text}")
    ])

    chain = prompt | llm | parser
    segments = chunk_text(text)
    collected = []

    for idx, seg in enumerate(segments, start=1):
        with st.spinner(f"分析功能片段 {idx}/{len(segments)}"):
            try:
                input_payload = {
                    "segment_id": f"seg_{idx}",
                    "segment_text": seg,
                    "format_instructions": parser.get_format_instructions()
                }
                if debug and log_fn:
                    log_fn({
                        "call": "analyze_features",
                        "phase": "input",
                        "segment_id": input_payload.get("segment_id"),
                        "payload": input_payload
                    })

                result = chain.invoke(input_payload)

                if debug and log_fn:
                    out_value = result if not hasattr(result, 'model_dump') else result.model_dump()
                    log_fn({
                        "call": "analyze_features",
                        "phase": "output",
                        "segment_id": input_payload.get("segment_id"),
                        "payload": out_value
                    })

                collected.extend(_normalize_feature_response(result))
            except Exception as err:
                st.warning(f"片段 {idx} 提取失败：{err}")

    # 去重：module + feature 作为 key
    unique = {}
    for feature in collected:
        key = (feature.module.strip(), feature.feature.strip())
        if key not in unique:
            unique[key] = feature
    return list(unique.values())


def generate_cases(llm, features, max_cases, debug: bool = False, log_fn=None):
    parser = JsonOutputParser(pydantic_object=TestSuite)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是测试架构师，根据功能定义设计测试用例，覆盖正向、异常、边界。"
                   "每条用例需要 case_id/module/feature/title/precondition/"
                   "steps/expected/priority/type。\n{format_instructions}"),
        ("human", "功能信息：\n{feature_payload}\n"
                  "请输出不超过 {max_cases} 条代表性测试用例。")
    ])

    chain = prompt | llm | parser
    all_cases = []

    def _normalize_case_response(result, fallback_module, fallback_feature):
        if result is None:
            return []

        raw_cases = []
        if hasattr(result, "cases"):
            raw_cases = getattr(result, "cases", [])
        elif isinstance(result, dict):
            raw_cases = result.get("cases", [])
        elif isinstance(result, list):
            raw_cases = result

        normalized = []
        for case in raw_cases:
            if isinstance(case, TestCase):
                data = case.model_dump()
            elif isinstance(case, dict):
                try:
                    data = TestCase(**case).model_dump()
                except ValidationError:
                    continue
            else:
                continue

            # 确保 module/feature 填充
            data.setdefault("module", fallback_module)
            data.setdefault("feature", fallback_feature)
            normalized.append(data)
        return normalized

    for idx, feature in enumerate(features, start=1):
        payload = feature.model_dump()
        try:
            with st.spinner(f"生成测试用例 {idx}/{len(features)}"):
                input_payload = {
                    "feature_payload": payload,
                    "max_cases": max_cases,
                    "format_instructions": parser.get_format_instructions()
                }
                if debug and log_fn:
                    log_fn({
                        "call": "generate_cases",
                        "phase": "input",
                        "feature": payload.get("module"),
                        "payload": input_payload
                    })

                result = chain.invoke(input_payload)

                if debug and log_fn:
                    out_value = result if not hasattr(result, 'model_dump') else result.model_dump()
                    log_fn({
                        "call": "generate_cases",
                        "phase": "output",
                        "feature": payload.get("module"),
                        "payload": out_value
                    })

                all_cases.extend(
                    _normalize_case_response(result, feature.module, feature.feature)
                )
        except Exception as err:
            st.warning(f"功能「{feature.feature}」生成失败：{err}")

    return all_cases


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Agent · 需求到测试用例", layout="wide")
st.title("🤖 Agent 流程：需求理解 ➜ 功能梳理 ➜ 测试用例")

with st.sidebar:
    st.header("LLM 配置")
    api_key = st.text_input("API Key", type="password")
    base_url = st.text_input("Base URL (可选)")
    model_name = st.selectbox(
        "模型",
        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "deepseek-chat"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_cases = st.slider("每个功能的最大用例数", 2, 36, 3)

    st.markdown("---")
    st.caption("支持上传 PDF / DOCX / TXT，也可以直接粘贴需求文本。")
    debug = st.checkbox("调试模式：显示每次 LLM 交互的输入/输出", key="debug")


# 根据调试开关拆分主视图：勾选时为两栏布局（左：主流程，右：固定调试日志）；未勾选时不显示调试区域
if debug:
    left_col, right_col = st.columns([1, 1])
    main = left_col
    right_col.markdown("### 调试输出")
    # 清空日志按钮
    if right_col.button("清空日志", key="clear_logs"):
        st.session_state['debug_logs'] = []

    # 下载日志 (.log)
    def _build_log_text():
        logs = st.session_state.get('debug_logs', [])
        sep = "\n" + ("-" * 60) + "\n"
        parts = []
        for e in logs:
            try:
                parts.append(sep + json.dumps(e, ensure_ascii=False, indent=2) + sep)
            except Exception:
                parts.append(sep + repr(e) + sep)
        return "\n".join(parts)

    right_col.download_button(
        "下载日志 (.log)",
        data=_build_log_text().encode("utf-8"),
        file_name="llm_debug.log",
        mime="text/plain",
        key="download_logs",
    )

    # debug_parent 用于渲染更复杂的 debug UI（分组 expander 列表）
    debug_parent = right_col
else:
    main = st
    debug_parent = None

uploaded = main.file_uploader("上传需求文档 (PDF/DOCX/TXT/MD)", type=["pdf", "docx", "txt", "md"])
text_input = main.text_area("或直接粘贴需求内容", height=200)

# 调试输出数据存储（始终保留），但仅在 debug=True 时渲染
if 'debug_logs' not in st.session_state:
    st.session_state['debug_logs'] = []

def render_debug_ui(parent):
    """在给定的列/容器中渲染按调用类型分组的日志（最新在前）。

    parent: DeltaGenerator（例如右侧列）
    """
    if parent is None:
        return

    # 新建一个占位区域来重新渲染
    display = parent.container()
    logs = list(reversed(st.session_state.get('debug_logs', [])))  # 最新在前

    # 按调用类型分组
    groups = {}
    for idx, entry in enumerate(logs):
        call = entry.get('call', 'other') if isinstance(entry, dict) else 'other'
        groups.setdefault(call, []).append((idx, entry))

    for call, items in groups.items():
        display.subheader(f"{call} ({len(items)})")
        for i, entry in items:
            header = f"{entry.get('ts','')} — {entry.get('phase', '')}"
            with display.expander(header, expanded=False):
                try:
                    display.json(entry)
                except Exception:
                    display.text(repr(entry))


def append_debug(entry):
    """追加一条结构化日志到 `st.session_state['debug_logs']` 并在 debug 时用 `st.json` 渲染显示。

    entry 可以是 dict（推荐）或任意可序列化对象。
    """
    try:
        logs = st.session_state.setdefault('debug_logs', [])
        ts = datetime.utcnow().isoformat() + "Z"
        if isinstance(entry, dict):
            entry_obj = {"ts": ts, **entry}
        else:
            entry_obj = {"ts": ts, "message": entry}

        logs.append(entry_obj)
        if len(logs) > 200:
            st.session_state['debug_logs'] = logs[-200:]

        # 仅在 debug 时更新右侧显示（用分组 expander 列表）
        if debug and debug_parent is not None:
            try:
                render_debug_ui(debug_parent)
            except Exception:
                try:
                    debug_parent.text("\n".join([repr(x) for x in st.session_state['debug_logs']]))
                except Exception:
                    pass
    except Exception:
        if debug and debug_parent is not None:
            try:
                debug_parent.text(repr(entry))
            except Exception:
                pass

# 页面初始渲染：如果已开启 debug，则渲染现有日志
if debug and debug_parent is not None:
    try:
        render_debug_ui(debug_parent)
    except Exception:
        try:
            debug_parent.text("\n".join([repr(x) for x in st.session_state.get('debug_logs', [])]))
        except Exception:
            pass

document_text = ""
if uploaded:
    try:
        document_text = extract_text(uploaded)
        main.success(f"文档解析成功，字符数：{len(document_text)}")
        with main.expander("查看文档内容"):
            main.text(document_text[:4000] + ("..." if len(document_text) > 4000 else ""))
    except Exception as exc:
        main.error(f"文件解析失败：{exc}")

elif text_input.strip():
    document_text = text_input.strip()


if main.button("🚀 运行 Agent 流程", type="primary"):
    if not document_text:
        main.error("请先上传文件或粘贴需求内容。")
    else:
        llm = init_llm(api_key, base_url, model_name, temperature)
        if llm:
            main.subheader("步骤 1：功能梳理")
            features = analyze_features(llm, document_text, debug=debug, log_fn=append_debug)

            if not features:
                main.error("没有提取到功能点，请检查文档内容。")
            else:
                feature_df = pd.DataFrame([f.dict() for f in features])
                main.dataframe(feature_df, use_container_width=True)

                feature_out = BytesIO()
                with pd.ExcelWriter(feature_out, engine="xlsxwriter") as writer:
                    feature_df.to_excel(writer, index=False, sheet_name="Features")

                main.download_button(
                    "📥 下载功能清单",
                    feature_out.getvalue(),
                    "features.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                main.subheader("步骤 2：自动生成测试用例")
                cases = generate_cases(llm, features, max_cases, debug=debug, log_fn=append_debug)

                if cases:
                    case_df = pd.DataFrame(cases)
                    main.dataframe(case_df, use_container_width=True)

                    case_out = BytesIO()
                    with pd.ExcelWriter(case_out, engine="xlsxwriter") as writer:
                        case_df.to_excel(writer, index=False, sheet_name="TestCases")

                    main.download_button(
                        "📥 下载测试用例",
                        case_out.getvalue(),
                        "test_cases.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    main.warning("LLM 没有返回测试用例，请尝试减少文档长度或更换模型。")

