import re
from io import BytesIO
from typing import List

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


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()


# 初始化 session state
for key in ["feature_df", "feature_bytes", "case_df", "case_bytes"]:
    st.session_state.setdefault(key, None)


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


def analyze_features(llm, text):
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
                result = chain.invoke({
                    "segment_id": f"seg_{idx}",
                    "segment_text": seg,
                    "format_instructions": parser.get_format_instructions()
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


def estimate_case_target(feature: FeatureItem, max_cases: int):
    """
    根据功能复杂度估算需要的测试用例数量。
    简单启发式：基于描述长度、验收点数以及依赖个数。
    """
    base = 1
    desc_len = len(feature.description.split())
    acceptance_count = len(feature.acceptance)
    dependency_count = len(feature.dependencies)

    if desc_len > 80:
        base += 2
    elif desc_len > 40:
        base += 1

    base += min(acceptance_count, 3)
    base += min(dependency_count, 2)

    return max(1, min(max_cases, base))


def generate_cases(llm, features, max_cases):
    parser = JsonOutputParser(pydantic_object=TestSuite)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是测试架构师，根据功能定义设计测试用例，覆盖正向、异常、边界。"
                   "每条用例需要 case_id/module/feature/title/precondition/"
                   "steps/expected/priority/type。你应根据功能复杂度，生成合适数量的用例，"
                   "但不要少于 1 条，也不要超过用户指定的上限。\n{format_instructions}"),
        ("human", "功能信息：\n{feature_payload}\n"
                  "请根据复杂度生成 {target_cases}~{max_cases} 条代表性用例，"
                  "若功能简单可输出更少，但至少 1 条。")
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
                target_cases = estimate_case_target(feature, max_cases)
                result = chain.invoke({
                    "feature_payload": payload,
                    "target_cases": target_cases,
                    "max_cases": max_cases,
                    "format_instructions": parser.get_format_instructions()
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


uploaded = st.file_uploader("上传需求文档 (PDF/DOCX/TXT/MD)", type=["pdf", "docx", "txt", "md"])
text_input = st.text_area("或直接粘贴需求内容", height=200)

document_text = ""
if uploaded:
    try:
        document_text = extract_text(uploaded)
        st.success(f"文档解析成功，字符数：{len(document_text)}")
        with st.expander("查看文档内容"):
            st.text(document_text[:4000] + ("..." if len(document_text) > 4000 else ""))
    except Exception as exc:
        st.error(f"文件解析失败：{exc}")

elif text_input.strip():
    document_text = text_input.strip()


def render_feature_results():
    df = st.session_state.get("feature_df")
    data = st.session_state.get("feature_bytes")
    if df is None:
        return
    st.subheader("步骤 1：功能梳理")
    st.dataframe(df, use_container_width=True)
    if data:
        st.download_button(
            "📥 下载功能清单",
            data=data,
            file_name="features.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_features_btn"
        )


def render_case_results():
    df = st.session_state.get("case_df")
    data = st.session_state.get("case_bytes")
    if df is None:
        return
    st.subheader("步骤 2：自动生成测试用例")
    st.dataframe(df, use_container_width=True)
    if data:
        st.download_button(
            "📥 下载测试用例",
            data=data,
            file_name="test_cases.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_cases_btn"
        )


if st.button("🚀 运行 Agent 流程", type="primary"):
    st.session_state["feature_df"] = None
    st.session_state["feature_bytes"] = None
    st.session_state["case_df"] = None
    st.session_state["case_bytes"] = None

    if not document_text:
        st.error("请先上传文件或粘贴需求内容。")
    else:
        llm = init_llm(api_key, base_url, model_name, temperature)
        if llm:
            features = analyze_features(llm, document_text)

            if not features:
                st.error("没有提取到功能点，请检查文档内容。")
            else:
                feature_df = pd.DataFrame([f.dict() for f in features])
                st.session_state["feature_df"] = feature_df
                st.session_state["feature_bytes"] = df_to_excel_bytes(feature_df, "Features")

                cases = generate_cases(llm, features, max_cases)

                if cases:
                    case_df = pd.DataFrame(cases)
                    st.session_state["case_df"] = case_df
                    st.session_state["case_bytes"] = df_to_excel_bytes(case_df, "TestCases")
                else:
                    st.warning("LLM 没有返回测试用例，请尝试减少文档长度或更换模型。")


render_feature_results()
render_case_results()

