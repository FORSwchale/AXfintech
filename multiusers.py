"""멀티유저/멀티세션 RAG 챗봇 — Supabase 사용자·세션 저장 + 벡터 검색."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"

load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = "gpt-4o-mini"
EMBED_BATCH_SIZE = 10
VECTOR_MATCH_COUNT = 10
PBKDF2_ITERATIONS = 260_000

CONFIG_KEYS = ("OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY")


# ---------------------------------------------------------------------------
# Config (st.secrets → .env)
# ---------------------------------------------------------------------------
def get_config_value(key: str) -> str:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:  # noqa: BLE001
        pass
    return os.getenv(key, "").strip()


def get_env_keys() -> dict[str, str]:
    return {k: get_config_value(k) for k in CONFIG_KEYS}


def check_missing_keys(keys: dict[str, str]) -> list[str]:
    return [k for k, v in keys.items() if not v]


# ---------------------------------------------------------------------------
# Logging (Streamlit Cloud safe)
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    log_candidates = [
        LOG_DIR,
        Path(tempfile.gettempdir()) / "multiusers_logs",
    ]
    for log_dir in log_candidates:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            probe = log_dir / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            log_path = log_dir / f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.WARNING)
            fh.setFormatter(fmt)
            root.addHandler(fh)
            break
        except (PermissionError, OSError):
            continue

    for name in (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "langchain",
        "langchain_openai",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multiusers_chatbot")


logger = _setup_logging()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""

APP_CSS = """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
"""


def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def get_supabase_client(keys: dict[str, str]) -> Client | None:
    if not keys["SUPABASE_URL"] or not keys["SUPABASE_ANON_KEY"]:
        return None
    return create_client(keys["SUPABASE_URL"], keys["SUPABASE_ANON_KEY"])


def get_llm(openai_key: str, temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        api_key=openai_key,
    )


def get_embeddings(openai_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(api_key=openai_key)


# ---------------------------------------------------------------------------
# Auth (PBKDF2-SHA256)
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return base64.b64encode(salt + dk).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        decoded = base64.b64decode(password_hash.encode("ascii"))
    except Exception:  # noqa: BLE001
        return False
    if len(decoded) < 17:
        return False
    salt, stored = decoded[:16], decoded[16:]
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return dk == stored


def signup_user(client: Client, login_id: str, password: str) -> dict[str, Any]:
    login_id = login_id.strip()
    if not login_id:
        raise ValueError("아이디를 입력해 주세요.")
    if not password:
        raise ValueError("비밀번호를 입력해 주세요.")
    if len(password) < 4:
        raise ValueError("비밀번호는 4자 이상이어야 합니다.")

    existing = (
        client.table("user")
        .select("id")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        raise ValueError("이미 사용 중인 아이디입니다.")

    row = {
        "login_id": login_id,
        "password_hash": hash_password(password),
    }
    resp = client.table("user").insert(row).execute()
    data = resp.data or []
    if not data:
        raise RuntimeError("회원가입에 실패했습니다.")
    user = data[0]
    return {"id": user["id"], "login_id": user["login_id"]}


def login_user(client: Client, login_id: str, password: str) -> dict[str, Any] | None:
    login_id = login_id.strip()
    if not login_id or not password:
        return None

    resp = (
        client.table("user")
        .select("id, login_id, password_hash")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None

    row = rows[0]
    if not verify_password(password, row["password_hash"]):
        return None
    return {"id": row["id"], "login_id": row["login_id"]}


def get_current_user_id() -> str:
    user = st.session_state.logged_in_user
    if not user:
        raise RuntimeError("로그인이 필요합니다.")
    return user["id"]


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------
def _format_memory_block(messages: list[dict[str, str]], max_items: int = 50) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_rag_messages(
    question: str,
    context: str,
    memory_text: str,
) -> list[SystemMessage | HumanMessage]:
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def _generate_followup_section(llm: ChatOpenAI, user_q: str, answer: str) -> str:
    trimmed = answer[:8000]
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(out, "content", str(out)) or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Follow-up generation failed: %s", exc)
        return ""

    raw = remove_separators(str(raw))
    if not raw.strip():
        return ""
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"


def generate_session_title(
    llm: ChatOpenAI,
    first_question: str,
    first_answer: str,
) -> str:
    prompt = (
        "다음 첫 질문과 답변을 15자 이내 한국어 세션 제목 한 줄로 요약하세요.\n"
        "따옴표, 설명, 번호 없이 제목만 출력하세요.\n\n"
        f"[질문]\n{first_question[:500]}\n\n[답변]\n{first_answer[:1500]}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        title = remove_separators(str(getattr(out, "content", "") or "")).strip()
        if title:
            return title[:80]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Session title generation failed: %s", exc)
    fallback = first_question.strip().replace("\n", " ")
    return (fallback[:30] + "...") if len(fallback) > 30 else (fallback or "새 세션")


# ---------------------------------------------------------------------------
# Supabase (user-scoped)
# ---------------------------------------------------------------------------
def list_sessions(client: Client, user_id: str) -> list[dict[str, Any]]:
    resp = (
        client.table("chat_sessions")
        .select("id, title, processed_files, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return resp.data or []


def fetch_session_messages(
    client: Client,
    session_id: str,
    user_id: str,
) -> list[dict[str, str]]:
    resp = (
        client.table("chat_messages")
        .select("role, content, msg_order")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("msg_order")
        .execute()
    )
    rows = resp.data or []
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def fetch_vector_file_names(
    client: Client,
    session_id: str,
    user_id: str,
) -> list[str]:
    resp = (
        client.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    names = sorted({row["file_name"] for row in (resp.data or []) if row.get("file_name")})
    return names


def delete_session_from_db(
    client: Client,
    session_id: str,
    user_id: str,
) -> None:
    client.table("chat_sessions").delete().eq("id", session_id).eq(
        "user_id", user_id
    ).execute()


def upsert_session_messages(
    client: Client,
    session_id: str,
    user_id: str,
    messages: list[dict[str, str]],
) -> None:
    client.table("chat_messages").delete().eq("session_id", session_id).eq(
        "user_id", user_id
    ).execute()
    if not messages:
        return
    rows = [
        {
            "user_id": user_id,
            "session_id": session_id,
            "role": m["role"],
            "content": m["content"],
            "msg_order": idx,
        }
        for idx, m in enumerate(messages)
    ]
    client.table("chat_messages").insert(rows).execute()


def save_session_to_db(
    client: Client,
    *,
    user_id: str,
    session_id: str,
    title: str,
    messages: list[dict[str, str]],
    processed_files: list[str],
    insert_new: bool,
) -> None:
    now = datetime.utcnow().isoformat()
    payload = {
        "id": session_id,
        "user_id": user_id,
        "title": title,
        "processed_files": processed_files,
        "updated_at": now,
    }
    if insert_new:
        payload["created_at"] = now
        client.table("chat_sessions").insert(payload).execute()
    else:
        client.table("chat_sessions").update(
            {
                "title": title,
                "processed_files": processed_files,
                "updated_at": now,
            }
        ).eq("id", session_id).eq("user_id", user_id).execute()

    upsert_session_messages(client, session_id, user_id, messages)


def store_vectors_for_file(
    client: Client,
    *,
    user_id: str,
    session_id: str,
    file_name: str,
    documents: list[Document],
    embeddings: OpenAIEmbeddings,
) -> int:
    if not documents:
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(documents)
    if not splits:
        return 0

    stored = 0
    for i in range(0, len(splits), EMBED_BATCH_SIZE):
        batch = splits[i : i + EMBED_BATCH_SIZE]
        texts = [doc.page_content for doc in batch]
        vectors = embeddings.embed_documents(texts)
        rows = []
        for doc, vector in zip(batch, vectors, strict=True):
            rows.append(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "file_name": file_name,
                    "content": doc.page_content,
                    "metadata": {
                        **(doc.metadata or {}),
                        "file_name": file_name,
                        "session_id": session_id,
                        "user_id": user_id,
                    },
                    "embedding": vector,
                }
            )
        client.table("vector_documents").insert(rows).execute()
        stored += len(rows)
    return stored


def process_pdf_uploads(
    client: Client,
    *,
    user_id: str,
    session_id: str,
    uploaded_files: list[Any],
    openai_key: str,
) -> list[str]:
    if not uploaded_files:
        return []

    embeddings = get_embeddings(openai_key)
    processed: list[str] = []

    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata = {**(doc.metadata or {}), "file_name": uf.name}
            store_vectors_for_file(
                client,
                user_id=user_id,
                session_id=session_id,
                file_name=uf.name,
                documents=docs,
                embeddings=embeddings,
            )
            processed.append(uf.name)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return processed


def search_vectors_rpc(
    client: Client,
    *,
    user_id: str,
    session_id: str,
    query: str,
    openai_key: str,
    match_count: int = VECTOR_MATCH_COUNT,
) -> list[Document]:
    embeddings = get_embeddings(openai_key)
    query_embedding = embeddings.embed_query(query)

    try:
        resp = client.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "filter_session_id": session_id,
                "filter_user_id": user_id,
            },
        ).execute()
        rows = resp.data or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("RPC vector search failed, fallback to table scan: %s", exc)
        resp = (
            client.table("vector_documents")
            .select("content, metadata, file_name")
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .limit(match_count)
            .execute()
        )
        rows = resp.data or []

    docs: list[Document] = []
    for row in rows:
        docs.append(
            Document(
                page_content=row.get("content", "") or "",
                metadata={
                    **(row.get("metadata") or {}),
                    "file_name": row.get("file_name"),
                },
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def get_active_session_id() -> str:
    if st.session_state.current_session_id:
        return st.session_state.current_session_id
    if not st.session_state.working_session_id:
        st.session_state.working_session_id = str(uuid.uuid4())
    return st.session_state.working_session_id


def reset_working_screen() -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.processed_names = []
    st.session_state.current_session_id = None
    st.session_state.working_session_id = str(uuid.uuid4())
    st.session_state.session_saved = False
    st.session_state.selected_session_id = None


def load_session_into_state(
    client: Client,
    session_id: str,
    user_id: str,
) -> None:
    messages = fetch_session_messages(client, session_id, user_id)
    sessions = list_sessions(client, user_id)
    title_map = {s["id"]: s for s in sessions}
    session_row = title_map.get(session_id, {})
    processed = session_row.get("processed_files") or []
    if not processed:
        processed = fetch_vector_file_names(client, session_id, user_id)

    st.session_state.chat_history = list(messages)
    st.session_state.conversation_memory = list(messages[-50:])
    st.session_state.processed_names = list(processed)
    st.session_state.current_session_id = session_id
    st.session_state.working_session_id = session_id
    st.session_state.session_saved = True
    st.session_state.selected_session_id = session_id


def auto_save_session(
    client: Client,
    openai_key: str,
    user_id: str,
) -> None:
    messages = st.session_state.chat_history
    if not messages:
        return

    llm = get_llm(openai_key, temperature=0.3)
    first_q = next((m["content"] for m in messages if m["role"] == "user"), "")
    first_a = next((m["content"] for m in messages if m["role"] == "assistant"), "")

    session_id = get_active_session_id()
    insert_new = not st.session_state.session_saved

    if insert_new and first_q and first_a:
        title = generate_session_title(llm, first_q, first_a)
    elif st.session_state.current_session_id:
        sessions = list_sessions(client, user_id)
        existing = next(
            (s for s in sessions if s["id"] == st.session_state.current_session_id),
            None,
        )
        title = (existing or {}).get("title") or "저장된 세션"
    else:
        title = generate_session_title(llm, first_q, first_a) if first_q else "새 세션"

    save_session_to_db(
        client,
        user_id=user_id,
        session_id=session_id,
        title=title,
        messages=messages,
        processed_files=st.session_state.processed_names,
        insert_new=insert_new,
    )
    st.session_state.current_session_id = session_id
    st.session_state.working_session_id = session_id
    st.session_state.session_saved = True


def manual_save_session(
    client: Client,
    openai_key: str,
    user_id: str,
) -> str:
    messages = st.session_state.chat_history
    if not messages:
        return "저장할 대화가 없습니다."

    llm = get_llm(openai_key, temperature=0.3)
    first_q = next((m["content"] for m in messages if m["role"] == "user"), "질문 없음")
    first_a = next(
        (m["content"] for m in messages if m["role"] == "assistant"),
        "답변 없음",
    )
    title = generate_session_title(llm, first_q, first_a)

    new_session_id = str(uuid.uuid4())
    old_session_id = get_active_session_id()

    save_session_to_db(
        client,
        user_id=user_id,
        session_id=new_session_id,
        title=title,
        messages=messages,
        processed_files=st.session_state.processed_names,
        insert_new=True,
    )

    if old_session_id != new_session_id and st.session_state.processed_names:
        resp = (
            client.table("vector_documents")
            .select("file_name, content, metadata, embedding")
            .eq("session_id", old_session_id)
            .eq("user_id", user_id)
            .execute()
        )
        rows = resp.data or []
        if rows:
            copied = []
            for row in rows:
                copied.append(
                    {
                        "user_id": user_id,
                        "session_id": new_session_id,
                        "file_name": row["file_name"],
                        "content": row["content"],
                        "metadata": row.get("metadata") or {},
                        "embedding": row["embedding"],
                    }
                )
            for i in range(0, len(copied), EMBED_BATCH_SIZE):
                client.table("vector_documents").insert(
                    copied[i : i + EMBED_BATCH_SIZE]
                ).execute()

    st.session_state.current_session_id = new_session_id
    st.session_state.working_session_id = new_session_id
    st.session_state.session_saved = True
    st.session_state.selected_session_id = new_session_id
    return f"세션이 저장되었습니다: {title}"


def _init_session() -> None:
    defaults = {
        "logged_in_user": None,
        "chat_history": [],
        "conversation_memory": [],
        "processed_names": [],
        "current_session_id": None,
        "working_session_id": str(uuid.uuid4()),
        "session_saved": False,
        "selected_session_id": None,
        "sessions_cache": [],
        "last_loaded_session_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_header(title_html: str) -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(title_html, unsafe_allow_html=True)
    with c3:
        st.empty()


def render_missing_keys(missing: list[str]) -> None:
    st.error(
        "# 환경변수 누락\n\n"
        "다음 키를 설정해 주세요: "
        f"**{', '.join(missing)}**\n\n"
        f"- 로컬: `{ENV_PATH}` 또는 `.env`\n"
        "- Streamlit Cloud: **Settings → Secrets**"
    )


def render_auth_screen(client: Client | None, keys: dict[str, str]) -> None:
    render_header(
        """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">숭실대학교</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
"""
    )

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown(
            """
<div style="text-align:center; margin-bottom:1rem;">
  <p style="color:#666;">로그인 또는 회원가입 후 PDF 기반 RAG 챗봇을 이용하세요.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

        with tab_login:
            login_id = st.text_input("아이디", key="login_id_input")
            login_pw = st.text_input("비밀번호", type="password", key="login_pw_input")
            if st.button("로그인", use_container_width=True, key="login_btn"):
                if client is None:
                    st.warning("Supabase 설정이 필요합니다.")
                elif not login_id or not login_pw:
                    st.warning("아이디와 비밀번호를 입력해 주세요.")
                else:
                    try:
                        user = login_user(client, login_id, login_pw)
                        if user is None:
                            st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
                        else:
                            st.session_state.logged_in_user = user
                            reset_working_screen()
                            st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Login failed: %s", exc)
                        st.error(f"로그인 중 오류가 발생했습니다: {exc}")

        with tab_signup:
            signup_id = st.text_input("아이디", key="signup_id_input")
            signup_pw = st.text_input("비밀번호", type="password", key="signup_pw_input")
            signup_pw2 = st.text_input(
                "비밀번호 확인", type="password", key="signup_pw2_input"
            )
            if st.button("회원가입", use_container_width=True, key="signup_btn"):
                if client is None:
                    st.warning("Supabase 설정이 필요합니다.")
                elif signup_pw != signup_pw2:
                    st.warning("비밀번호가 일치하지 않습니다.")
                else:
                    try:
                        user = signup_user(client, signup_id, signup_pw)
                        st.session_state.logged_in_user = user
                        reset_working_screen()
                        st.success("회원가입이 완료되었습니다. 대시보드로 이동합니다.")
                        st.rerun()
                    except ValueError as exc:
                        st.warning(str(exc))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Signup failed: %s", exc)
                        st.error(f"회원가입 중 오류가 발생했습니다: {exc}")


def render_sidebar(
    client: Client | None,
    keys: dict[str, str],
    user_id: str,
) -> None:
    user = st.session_state.logged_in_user or {}
    with st.sidebar:
        st.markdown(f"**로그인:** `{user.get('login_id', '-')}`")

        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in_user = None
            reset_working_screen()
            st.rerun()

        st.markdown("---")
        st.markdown("### 세션 관리")

        sessions: list[dict[str, Any]] = []
        if client is not None:
            try:
                sessions = list_sessions(client, user_id)
                st.session_state.sessions_cache = sessions
            except Exception as exc:  # noqa: BLE001
                logger.warning("Session list failed: %s", exc)
                st.error(f"세션 목록을 불러오지 못했습니다: {exc}")

        session_options = ["(새 세션)"] + [
            f"{s.get('title', '제목 없음')} ({s['id'][:8]})" for s in sessions
        ]
        session_ids = [None] + [s["id"] for s in sessions]

        current_index = 0
        if st.session_state.selected_session_id in session_ids:
            current_index = session_ids.index(st.session_state.selected_session_id)

        selected_label = st.selectbox(
            "세션 선택",
            session_options,
            index=current_index,
            key="session_selectbox",
        )
        selected_idx = session_options.index(selected_label)
        selected_id = session_ids[selected_idx]

        if selected_id != st.session_state.selected_session_id:
            st.session_state.selected_session_id = selected_id
            if selected_id is None:
                reset_working_screen()
                st.rerun()
            elif client is not None:
                try:
                    load_session_into_state(client, selected_id, user_id)
                    st.session_state.last_loaded_session_id = selected_id
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Auto session load failed: %s", exc)
                    st.error(f"세션 자동 로드 실패: {exc}")

        btn_cols = st.columns(2)
        with btn_cols[0]:
            if st.button("세션저장", use_container_width=True):
                if client is None:
                    st.warning("Supabase 설정이 필요합니다.")
                elif not keys["OPENAI_API_KEY"]:
                    st.warning("OPENAI_API_KEY가 필요합니다.")
                else:
                    try:
                        msg = manual_save_session(client, keys["OPENAI_API_KEY"], user_id)
                        st.success(msg)
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Manual save failed: %s", exc)
                        st.error(f"세션 저장 실패: {exc}")

        with btn_cols[1]:
            if st.button("세션로드", use_container_width=True):
                if client is None:
                    st.warning("Supabase 설정이 필요합니다.")
                elif not st.session_state.selected_session_id:
                    st.warning("로드할 세션을 선택하세요.")
                else:
                    try:
                        load_session_into_state(
                            client, st.session_state.selected_session_id, user_id
                        )
                        st.success("세션을 불러왔습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Session load failed: %s", exc)
                        st.error(f"세션 로드 실패: {exc}")

        btn_cols2 = st.columns(2)
        with btn_cols2[0]:
            if st.button("세션삭제", use_container_width=True):
                if client is None:
                    st.warning("Supabase 설정이 필요합니다.")
                elif not st.session_state.selected_session_id:
                    st.warning("삭제할 세션을 선택하세요.")
                else:
                    try:
                        delete_session_from_db(
                            client, st.session_state.selected_session_id, user_id
                        )
                        if (
                            st.session_state.current_session_id
                            == st.session_state.selected_session_id
                        ):
                            reset_working_screen()
                        st.success("선택한 세션이 삭제되었습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Session delete failed: %s", exc)
                        st.error(f"세션 삭제 실패: {exc}")

        with btn_cols2[1]:
            if st.button("화면초기화", use_container_width=True):
                reset_working_screen()
                st.success("화면이 초기화되었습니다.")
                st.rerun()

        if st.button("vectordb", use_container_width=True):
            if client is None:
                st.warning("Supabase 설정이 필요합니다.")
            else:
                sid = get_active_session_id()
                try:
                    names = fetch_vector_file_names(client, sid, user_id)
                    if names:
                        st.info(
                            "현재 vectordb 파일:\n" + "\n".join(f"- {n}" for n in names)
                        )
                    else:
                        st.info("현재 vectordb에 저장된 파일이 없습니다.")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Vector file list failed: %s", exc)
                    st.error(f"vectordb 조회 실패: {exc}")

        st.markdown("---")
        st.markdown(f"**LLM 모델:** `{MODEL_NAME}`")

        uploads = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("파일 처리하기"):
            if client is None:
                st.warning("Supabase 설정이 필요합니다.")
            elif not keys["OPENAI_API_KEY"]:
                st.warning("OPENAI_API_KEY가 필요합니다.")
            elif not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            else:
                try:
                    sid = get_active_session_id()
                    names = process_pdf_uploads(
                        client,
                        user_id=user_id,
                        session_id=sid,
                        uploaded_files=list(uploads),
                        openai_key=keys["OPENAI_API_KEY"],
                    )
                    merged = list(dict.fromkeys(st.session_state.processed_names + names))
                    st.session_state.processed_names = merged
                    auto_save_session(client, keys["OPENAI_API_KEY"], user_id)
                    st.success("PDF 처리 및 세션 자동 저장이 완료되었습니다.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PDF processing failed: %s", exc)
                    st.error(f"PDF 처리 중 오류: {exc}")

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"- {name}")

        mem_count = len(st.session_state.conversation_memory)
        file_count = len(st.session_state.processed_names)
        sid = st.session_state.current_session_id or st.session_state.working_session_id
        settings_text = (
            f"모델: {MODEL_NAME}\n"
            f"현재 세션 ID: {sid[:8] if sid else '-'}\n"
            f"처리된 PDF 파일 수: {file_count}\n"
            f"대화 기록(메시지) 수: {mem_count}\n"
            f"저장 상태: {'저장됨' if st.session_state.session_saved else '미저장'}"
        )
        st.text(settings_text)


def render_dashboard(
    client: Client | None,
    keys: dict[str, str],
    user_id: str,
) -> None:
    render_header(
        """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">숭실대학교</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
"""
    )
    render_sidebar(client, keys, user_id)

    user = st.session_state.logged_in_user or {}
    if not st.session_state.chat_history:
        st.markdown(
            f"""
### 안녕하세요, **{user.get('login_id', '사용자')}**님!

PDF를 업로드하고 질문하면 문서 기반 RAG 답변을 받을 수 있습니다.
사이드바에서 세션을 저장·로드·삭제할 수 있으며, 대화와 PDF 처리 후 자동 저장됩니다.
"""
        )

    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = remove_separators(msg["content"])
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append({"role": "user", "content": user_input})
    if len(st.session_state.conversation_memory) > 50:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            llm = get_llm(keys["OPENAI_API_KEY"])
            session_id = get_active_session_id()
            has_vectors = bool(st.session_state.processed_names)

            if has_vectors and client is not None:
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                docs = search_vectors_rpc(
                    client,
                    user_id=user_id,
                    session_id=session_id,
                    query=user_input,
                    openai_key=keys["OPENAI_API_KEY"],
                )
                if docs:
                    context = "\n\n".join(d.page_content for d in docs)
                    messages = _build_rag_messages(user_input, context, mem_txt)
                else:
                    sys = (
                        f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n"
                        f"{mem_txt or '(없음)'}\n\n"
                        "업로드된 PDF에서 관련 내용을 찾지 못했습니다. 일반 지식 범위에서 답하되 한계를 밝히세요."
                    )
                    messages = [
                        SystemMessage(content=sys),
                        HumanMessage(content=user_input),
                    ]
            else:
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                sys = f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n{mem_txt or '(없음)'}"
                messages = [
                    SystemMessage(content=sys),
                    HumanMessage(content=user_input),
                ]

            acc = ""
            for chunk in llm.stream(messages):
                piece = getattr(chunk, "content", "") or ""
                if piece:
                    acc += piece
                    placeholder.markdown(remove_separators(acc) + "▌")
            full_answer = remove_separators(acc)
            placeholder.markdown(full_answer)

            follow = _generate_followup_section(llm, user_input, full_answer)
            if follow:
                full_answer += follow
                placeholder.markdown(remove_separators(full_answer))

        except Exception as exc:  # noqa: BLE001
            logger.warning("Answer generation failed: %s", exc)
            full_answer = (
                f"# 오류\n\n요청을 처리하는 중 문제가 발생했습니다.\n\n`{exc}`"
            )
            placeholder.markdown(remove_separators(full_answer))

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_answer}
        )
        st.session_state.conversation_memory.append(
            {"role": "assistant", "content": full_answer}
        )
        if len(st.session_state.conversation_memory) > 50:
            st.session_state.conversation_memory = st.session_state.conversation_memory[
                -50:
            ]

        if client is not None:
            try:
                auto_save_session(client, keys["OPENAI_API_KEY"], user_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Auto save failed: %s", exc)


def main() -> None:
    st.set_page_config(
        page_title="숭실대학교 RAG 챗봇",
        page_icon="📚",
        layout="wide",
    )
    _init_session()

    keys = get_env_keys()
    missing = check_missing_keys(keys)

    if missing:
        render_header(
            """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">숭실대학교</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
"""
        )
        render_missing_keys(missing)
        return

    client = get_supabase_client(keys)

    if not st.session_state.logged_in_user:
        render_auth_screen(client, keys)
        return

    user_id = get_current_user_id()
    render_dashboard(client, keys, user_id)


if __name__ == "__main__":
    main()
