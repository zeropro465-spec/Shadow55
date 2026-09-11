import streamlit as st
from groq import Groq
from ddgs import DDGS
import uuid
import base64

st.set_page_config(page_title="Shadow AI", page_icon="🤖", layout="centered")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Settings")
    user_api_key = st.text_input("Enter your Groq API Key:", type="password")
    st.markdown("---")

    st.subheader("💬 Chats")

    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        first_id = str(uuid.uuid4())[:8]
        st.session_state.chats[first_id] = []
        st.session_state.current_chat_id = first_id

    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.chats[new_id] = []
        st.session_state.current_chat_id = new_id
        st.rerun()

    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        st.session_state.chats[st.session_state.current_chat_id] = []
        st.rerun()

    st.markdown("---")
    st.markdown("### History")

    for chat_id in list(st.session_state.chats.keys()):
        msgs = st.session_state.chats[chat_id]
        if msgs and msgs[0].get("content"):
            title = msgs[0]["content"][:20] + ("..." if len(msgs[0]["content"]) > 20 else "")
        else:
            title = f"Chat {chat_id}"

        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            btn_type = "primary" if chat_id == st.session_state.current_chat_id else "secondary"
            if st.button(title, key=f"chat_btn_{chat_id}", use_container_width=True, type=btn_type):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with col2:
            if len(st.session_state.chats) > 1:
                if st.button("❌", key=f"del_btn_{chat_id}"):
                    del st.session_state.chats[chat_id]
                    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                    st.rerun()

    st.markdown("---")
    st.subheader("🔍 Options")
    use_web_search = st.toggle("Enable web search", value=False)
    
    # Yalnız tələb olunan model təyin edildi
    model_choice = "openai/gpt-oss-120b"
    st.text(f"Model: {model_choice}")

    st.markdown("---")
    st.markdown("💡 *Get a free Groq API key from [Groq Console](https://console.groq.com/).*")

# ---------- MAIN ----------
st.title("🤖 Shadow - Smart Assistant")
st.caption("An AI assistant with internet access and chat management")

if not user_api_key:
    st.info("👈 Please enter your Groq API key in the sidebar to begin.")
    st.stop()

try:
    client = Groq(api_key=user_api_key)
except Exception as e:
    st.error(f"Invalid API key: {e}")
    st.stop()

def web_search(query: str, max_results: int = 3) -> str:
    try:
        with DDGS() as ddgs:
            results = [r.get("body", "") for r in ddgs.text(query, max_results=max_results)]
        return " ".join(filter(None, results))
    except Exception as e:
        return f"[Search error: {e}]"

current_chat = st.session_state.current_chat_id
if current_chat not in st.session_state.chats:
    st.session_state.chats[current_chat] = []

for message in st.session_state.chats[current_chat]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

uploaded_file = st.file_uploader(
    "Upload a text or PDF file (optional):",
    type=["txt", "pdf"],
    key=f"uploader_{current_chat}",
)

if prompt := st.chat_input("What would you like to ask Shadow?"):
    file_bytes = None
    file_type = None
    extracted_text = ""

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_type = uploaded_file.type

        if file_type == "text/plain":
            try:
                extracted_text = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                extracted_text = ""
        elif file_type == "application/pdf":
            try:
                import io
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(file_bytes))
                extracted_text = "\n".join(
                    (page.extract_text() or "") for page in reader.pages
                )
            except Exception:
                extracted_text = "[PDF could not be read]"

    msg_data = {"role": "user", "content": prompt}
    st.session_state.chats[current_chat].append(msg_data)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("Shadow is thinking..."):
            search_data = ""
            if use_web_search:
                search_data = web_search(prompt)

            system_prompt = (
                "Your name is Shadow. You are a high-level analyst and AI assistant "
                "with deep reasoning capabilities.\n"
                "Important rule: Always respond in the exact language the user writes in."
            )
            if search_data:
                system_prompt += f"\n\n[Real-time web search data]:\n{search_data}"
            if extracted_text:
                system_prompt += f"\n\n[File content provided by user]:\n{extracted_text}"

            llm_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            try:
                stream = client.chat.completions.create(
                    model=model_choice,
                    messages=llm_messages,
                    temperature=0.5,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_response += delta
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"❌ An error occurred: {e}"
                placeholder.error(full_response)

        st.session_state.chats[current_chat].append(
            {"role": "assistant", "content": full_response}
        )
