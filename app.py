import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chat",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Premium dark CSS ──────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d0d14 0%, #111827 50%, #0d0d14 100%);
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: rgba(15, 15, 25, 0.95);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

.chat-header {
    text-align: center;
    padding: 2rem 0 1rem;
}
.chat-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc, #818cf8);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    margin-bottom: 0.25rem;
}
.chat-header p { color: #64748b; font-size: 0.9rem; }
@keyframes shimmer { to { background-position: 200% center; } }

[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    margin-bottom: 0.75rem !important;
    border: 1px solid rgba(99, 102, 241, 0.12) !important;
    backdrop-filter: blur(10px);
    transition: all 0.2s ease;
}
[data-testid="stChatMessage"]:hover {
    border-color: rgba(99, 102, 241, 0.3) !important;
}

[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(99, 102, 241, 0.8) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

.provider-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 3px 10px;
    border-radius: 999px;
    margin-top: 6px;
    text-transform: uppercase;
}
.badge-gemini {
    background: rgba(52, 211, 153, 0.15);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}
.badge-openai {
    background: rgba(99, 102, 241, 0.15);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.3);
}
.badge-error {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.token-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-top: 8px;
    font-size: 0.78rem;
    color: #94a3b8;
    display: flex;
    gap: 1.5rem;
}
.token-card span { color: #c084fc; font-weight: 600; }

.metric-box {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #94a3b8;
}
.metric-box .val { font-size: 1.3rem; font-weight: 700; color: #a5b4fc; }

hr { border-color: rgba(99, 102, 241, 0.15) !important; }

.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()

# ── Provider setup ────────────────────────────────────────────────────────────
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 20.0


@st.cache_resource
def get_providers():
    return [
        {
            "name": "gemini",
            "label": "Gemini 2.5 Flash",
            "model": "gemini-2.5-flash",
            "client": OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                max_retries=0,
                timeout=30,
            ),
        },
        {
            "name": "openai",
            "label": "GPT-4o Mini",
            "model": "gpt-4o-mini",
            "client": OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                max_retries=0,
                timeout=30,
            ),
        },
    ]


def call_with_retries(provider, status_placeholder=None, **params):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return provider["client"].chat.completions.create(
                model=provider["model"], **params
            )
        except RETRYABLE as e:
            if attempt == MAX_RETRIES:
                raise
            delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)
            delay += random.uniform(0, delay * 0.25)
            if status_placeholder:
                status_placeholder.warning(
                    f"⚡ [{provider['name']}] {type(e).__name__} — retry {attempt+1}/{MAX_RETRIES} in {delay:.1f}s…"
                )
            time.sleep(delay)


def chat_with_providers(providers, status_placeholder=None, **params):
    last_error = None
    for provider in providers:
        try:
            if status_placeholder:
                status_placeholder.info(f"🔄 Calling **{provider['label']}**…")
            response = call_with_retries(provider, status_placeholder, **params)
            return response, provider["name"]
        except Exception as e:
            if status_placeholder:
                status_placeholder.warning(
                    f"❌ {provider['label']} failed — trying next provider…"
                )
            last_error = e
    raise RuntimeError("All providers failed") from last_error


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✦ AI Chat")
    st.markdown("Multi-provider chatbot with automatic fallback.")
    st.divider()

    st.markdown("### ⚙️ Settings")
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, concise, and friendly AI assistant.",
        height=100,
        help="Instructions given to the AI before every conversation.",
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05,
                             help="Higher = more creative. Lower = more focused.")
    max_tokens = st.slider("Max tokens", 128, 4096, 1024, 64,
                            help="Maximum length of the AI response.")

    st.divider()
    st.markdown("### 📊 Session Stats")
    st.markdown(
        f"""
        <div class="metric-box"><div class="val">{st.session_state.total_messages}</div>Messages sent</div>
        <div class="metric-box"><div class="val">{st.session_state.total_tokens:,}</div>Total tokens used</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### 🔌 Providers")
    providers_info = [
        ("🟢", "Gemini 2.5 Flash", "gemini-2.5-flash", "Primary"),
        ("🔵", "GPT-4o Mini", "gpt-4o-mini", "Fallback"),
    ]
    for icon, label, model, role in providers_info:
        st.markdown(f"{icon} **{label}** `{model}` — *{role}*")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.session_state.total_messages = 0
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="chat-header">
        <h1>✦ AI Chat</h1>
        <p>Powered by Gemini 2.5 Flash &amp; GPT-4o Mini · automatic provider fallback</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "provider" in msg:
            badge_cls = f"badge-{msg['provider']}"
            icon = "✦" if msg["provider"] == "gemini" else "◈"
            label = "Gemini 2.5 Flash" if msg["provider"] == "gemini" else "GPT-4o Mini"
            st.markdown(
                f'<div class="provider-badge {badge_cls}">{icon} {label}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("usage"):
                u = msg["usage"]
                st.markdown(
                    f"""<div class="token-card">
                        Prompt&nbsp;<span>{u['prompt']}</span>
                        &nbsp;·&nbsp;Completion&nbsp;<span>{u['completion']}</span>
                        &nbsp;·&nbsp;Total&nbsp;<span>{u['total']}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask me anything…"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.total_messages += 1

    api_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        status = st.empty()
        try:
            providers = get_providers()
            response, used = chat_with_providers(
                providers,
                status_placeholder=status,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            status.empty()

            reply = response.choices[0].message.content
            st.markdown(reply)

            badge_cls = f"badge-{used}"
            icon = "✦" if used == "gemini" else "◈"
            label = "Gemini 2.5 Flash" if used == "gemini" else "GPT-4o Mini"
            st.markdown(
                f'<div class="provider-badge {badge_cls}">{icon} {label}</div>',
                unsafe_allow_html=True,
            )

            usage_data = None
            if response.usage:
                u = response.usage
                usage_data = {
                    "prompt": u.prompt_tokens,
                    "completion": u.completion_tokens,
                    "total": u.total_tokens,
                }
                st.markdown(
                    f"""<div class="token-card">
                        Prompt&nbsp;<span>{u.prompt_tokens}</span>
                        &nbsp;·&nbsp;Completion&nbsp;<span>{u.completion_tokens}</span>
                        &nbsp;·&nbsp;Total&nbsp;<span>{u.total_tokens}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.session_state.total_tokens += u.total_tokens

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": reply,
                    "provider": used,
                    "usage": usage_data,
                }
            )

        except Exception as e:
            status.empty()
            st.markdown(
                f'<div class="provider-badge badge-error">⚠ Error: {type(e).__name__}</div>',
                unsafe_allow_html=True,
            )
            st.error(str(e))
