import streamlit as st
from groq import Groq
from ddgs import DDGS
import uuid
import base64
import json
import io
import os
from datetime import datetime

# ---------- OPTIONAL IMPORTS (crash olmasın) ----------
try:
    from pypdf import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from streamlit_mic_recorder import speech_to_text
    HAS_STT = True
except ImportError:
    HAS_STT = False

try:
    from gtts import gTTS
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

try:
    import replicate
    HAS_REPLICATE = True
except ImportError:
    HAS_REPLICATE = False


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Shadow AI - Ultimate",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .tag { display:inline-block; background:#7C3AED; color:white;
           padding:2px 8px; border-radius:8px; font-size:12px; margin:2px; }
    .metric-box { background:#1E1E2E; padding:8px; border-radius:8px; }
</style>
""", unsafe_allow_html=True)


# ---------- SESSION STATE INIT ----------
defaults = {
    "chats": {},
    "current_chat_id": None,
    "tags": {},
    "personas": {},
    "prefill": "",
    "voice_input": "",
    "vector_docs": {},  # chat_id -> list of docs
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.current_chat_id is None:
    first_id = str(uuid.uuid4())[:8]
    st.session_state.chats[first_id] = []
    st.session_state.current_chat_id = first_id


# ---------- PERSONAS ----------
PERSONAS = {
    "🧠 Default": "You are Shadow, a helpful AI assistant. Respond in the user's language.",
    "👨‍💻 Developer": "You are an experienced software engineer. Provide code examples and technical depth. Respond in the user's language.",
    "📊 Analyst": "You are a business analyst. Provide data-driven, structured answers. Respond in the user's language.",
    "🎨 Creative": "You are a creative writer. Use vivid metaphors and storytelling. Respond in the user's language.",
    "🎓 Teacher": "You are a patient teacher. Explain step-by-step in simple terms. Respond in the user's language.",
}


# ---------- HELPERS ----------
def web_search(query: str, max_results: int = 3) -> str:
    try:
        with DDGS() as ddgs:
            results = [r.get("body", "") for r in ddgs.text(query, max_results=max_results)]
        return " ".join(filter(None, results))
    except Exception as e:
        return f"[Search error: {e}]"


def count_tokens(text: str) -> int:
    if not HAS_TIKTOKEN or not text:
        return 0
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return 0


def chat_to_markdown(messages) -> str:
    md = f"# Shadow AI Söhbəti\n\n_Tarix: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"
    for m in messages:
        role = "👤 **User**" if m["role"] == "user" else "🤖 **Shadow**"
        md += f"{role}:\n\n{m['content']}\n\n---\n\n"
    return md


def generate_title(first_message: str, client) -> str:
    try:
        r = client.chat.completions.create(
            messages=[{"role": "user", "content": f"3-5 sözlə qısa başlıq ver (yalnız başlıq): {first_message[:200]}"}],
            model="llama-3.1-8b-instant",
            temperature=0.3,
        )
        return r.choices[0].message.content.strip().strip('"').strip("'")[:40]
    except Exception:
        return first_message[:20] + "..."


def speak_text(text: str, lang: str = "az"):
    if not HAS_TTS:
        return None
    try:
        tts = gTTS(text=text[:3000], lang=lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception:
        return None


def extract_pdf_text(file_bytes: bytes) -> str:
    if not HAS_PDF:
        return "[pypdf quraşdırılmayıb]"
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        return f"[PDF xətası: {e}]"


def ocr_image(file_bytes: bytes) -> str:
    if not HAS_OCR:
        return ""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img, lang="aze+eng")
    except Exception:
        return ""


# ---------- CHROMA (RAG) ----------
@st.cache_resource
def get_chroma_client():
    if not HAS_CHROMA:
        return None
    try:
        return chromadb.Client()
    except Exception:
        return None


def add_to_vector_store(chat_id: str, text: str, doc_id: str):
    client_db = get_chroma_client()
    if client_db is None:
        return False
    try:
        coll = client_db.get_or_create_collection(f"chat_{chat_id}")
        coll.add(documents=[text], ids=[doc_id])
        return True
    except Exception:
        return False


def query_vector_store(chat_id: str, query: str, n: int = 3) -> str:
    client_db = get_chroma_client()
    if client_db is None:
        return ""
    try:
        coll = client_db.get_or_create_collection(f"chat_{chat_id}")
        if coll.count() == 0:
            return ""
        results = coll.query(query_texts=[query], n_results=min(n, coll.count()))
        return "\n\n".join(results.get("documents", [[]])[0])
    except Exception:
        return ""


# ============================================================
#                       SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    user_api_key = st.text_input("Groq API Key:", type="password")
    st.markdown("---")

    # ---------- PERSONA ----------
    st.subheader("🎭 Persona")
    persona_name = st.selectbox("Choose persona:", list(PERSONAS.keys()))

    # ---------- MODEL ----------
    st.subheader("🤖 Model")
    model_choice = st.selectbox(
        "Select model:",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "meta-llama/llama-4-scout-17b-16e-instruct",  # vision dəstəyi
        ],
        index=0,
    )

    # ---------- OPTIONS ----------
    st.subheader("🔍 Options")
    use_web_search = st.toggle("Enable web search", value=False)
    use_rag = st.toggle("Enable RAG (documents)", value=False)
    auto_title = st.toggle("Auto-generate titles", value=True)

    # ---------- VOICE ----------
    if HAS_STT:
        st.markdown("---")
        st.subheader("🎤 Voice Input")
        voice_text = speech_to_text(
            language="az",
            start_prompt="🎤 Start",
            stop_prompt="⏹ Stop",
            just_once=True,
            key="voice",
        )
        if voice_text:
            st.session_state.prefill = voice_text
            st.success(f"Alındı: {voice_text[:40]}...")

    # ---------- CHATS ----------
    st.markdown("---")
    st.subheader("💬 Chats")

    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.chats[new_id] = []
        st.session_state.current_chat_id = new_id
        st.rerun()

    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        st.session_state.chats[st.session_state.current_chat_id] = []
        st.rerun()

    # ---------- SEARCH IN CHATS ----------
    st.markdown("---")
    st.markdown("### 🔎 Search in Chats")
    search_query = st.text_input("Search text:", key="chat_search")
    if search_query:
        found = False
        for cid, msgs in st.session_state.chats.items():
            for i, m in enumerate(msgs):
                if search_query.lower() in m["content"].lower():
                    st.markdown(f"**{cid}** #{i}: ...{m['content'][:60]}...")
                    found = True
        if not found:
            st.caption("Nəticə tapılmadı.")

    # ---------- HISTORY ----------
    st.markdown("---")
    st.markdown("### 📚 History")

    for chat_id in list(st.session_state.chats.keys()):
        msgs = st.session_state.chats[chat_id]
        if msgs and msgs[0].get("content"):
            title = msgs[0]["content"][:20] + ("..." if len(msgs[0]["content"]) > 20 else "")
        else:
            title = f"Chat {chat_id}"

        # Tags göstər
        tags = st.session_state.tags.get(chat_id, [])

        col1, col2 = st.columns([0.75, 0.25])
        with col1:
            btn_type = "primary" if chat_id == st.session_state.current_chat_id else "secondary"
            if st.button(f"📌 {title}", key=f"chat_btn_{chat_id}",
                         use_container_width=True, type=btn_type):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with col2:
            if len(st.session_state.chats) > 1:
                if st.button("❌", key=f"del_btn_{chat_id}"):
                    del st.session_state.chats[chat_id]
                    st.session_state.tags.pop(chat_id, None)
                    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                    st.rerun()

        if tags:
            st.markdown(
                " ".join(f'<span class="tag">{t}</span>' for t in tags),
                unsafe_allow_html=True,
            )

    # ---------- TAGS ----------
    st.markdown("---")
    st.markdown("### 🏷️ Tags")
    new_tag = st.text_input("Add tag to current chat:", key="tag_input")
    if st.button("Add Tag") and new_tag.strip():
        st.session_state.tags.setdefault(st.session_state.current_chat_id, [])
        st.session_state.tags[st.session_state.current_chat_id].append(new_tag.strip())
        st.rerun()

    # ---------- EXPORT / IMPORT ----------
    st.markdown("---")
    st.markdown("### 💾 Export / Import")

    export_json = json.dumps(st.session_state.chats, default=str, ensure_ascii=False, indent=2)
    st.download_button(
        "📥 Download all chats (JSON)",
        data=export_json,
        file_name=f"shadow_chats_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
        use_container_width=True,
    )

    current_msgs = st.session_state.chats.get(st.session_state.current_chat_id, [])
    st.download_button(
        "📄 Download current chat (MD)",
        data=chat_to_markdown(current_msgs),
        file_name=f"chat_{st.session_state.current_chat_id}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    import_file = st.file_uploader("📤 Import chats (JSON)", type=["json"], key="import_json")
    if import_file:
        try:
            data = json.load(import_file)
            if isinstance(data, dict):
                st.session_state.chats.update(data)
                st.success(f"{len(data)} söhbət yükləndi.")
        except Exception as e:
            st.error(f"Xəta: {e}")

    st.markdown("---")
    st.markdown("💡 [Groq API key](https://console.groq.com/)")


# ============================================================
#                       MAIN PAGE
# ============================================================
st.title("🤖 Shadow AI - Ultimate")
st.caption("Groq + Web Search + RAG + Vision + Voice + RAG")

if not user_api_key:
    st.info("👈 Please enter your Groq API key in the sidebar.")
    st.stop()

try:
    client = Groq(api_key=user_api_key)
except Exception as e:
    st.error(f"Invalid API key: {e}")
    st.stop()

# ---------- UPLOAD SIDEBAR (docs for RAG) ----------
with st.expander("📚 RAG Documents (add to vector store)", expanded=False):
    if not HAS_CHROMA:
        st.warning("chromadb quraşdırılmayıb: `pip install chromadb`")
    else:
        rag_files = st.file_uploader(
            "Upload documents to index:",
            type=["txt", "pdf", "md"],
            accept_multiple_files=True,
            key="rag_upload",
        )
        if rag_files and st.button("Index Documents"):
            for f in rag_files:
                content = ""
                if f.type == "application/pdf":
                    content = extract_pdf_text(f.getvalue())
                else:
                    content = f.getvalue().decode("utf-8", errors="ignore")
                if content:
                    ok = add_to_vector_store(
                        st.session_state.current_chat_id,
                        content,
                        doc_id=f"{f.name}_{uuid.uuid4().hex[:6]}",
                    )
                    if ok:
                        st.success(f"✔ {f.name} indexed")
                    else:
                        st.error(f"✘ {f.name} failed")
        st.caption("Indexed docs are used when 'Enable RAG' is on.")


# ---------- CHAT STATE ----------
current_chat = st.session_state.current_chat_id
if current_chat not in st.session_state.chats:
    st.session_state.chats[current_chat] = []

# ---------- TOKEN COUNTER ----------
if HAS_TIKTOKEN:
    total_tokens = sum(count_tokens(m.get("content", "")) for m in st.session_state.chats[current_chat])
    colA, colB, colC = st.columns(3)
    colA.metric("💬 Messages", len(st.session_state.chats[current_chat]))
    colB.metric("🔢 Tokens (est.)", total_tokens)
    colC.metric("💰 Cost (est.)", f"${total_tokens * 0.0000005:.6f}")


# ---------- RENDER MESSAGES WITH ACTIONS ----------
for i, message in enumerate(st.session_state.chats[current_chat]):
    with st.chat_message(message["role"]):
        if message.get("image"):
            st.image(message["image"], width=300)
        st.markdown(message["content"])

        # Actions for assistant messages
        if message["role"] == "assistant":
            c1, c2, c3 = st.columns([1, 1, 6])
            if c1.button("🔁", key=f"regen_{i}", help="Regenerate"):
                # cut off this and following messages
                st.session_state.chats[current_chat] = st.session_state.chats[current_chat][:i]
                st.rerun()
            if c2.button("🗑️", key=f"del_{i}", help="Delete"):
                st.session_state.chats[current_chat].pop(i)
                st.rerun()
            if c3.button("🔊", key=f"tts_{i}", help="Read aloud") and HAS_TTS:
                audio = speak_text(message["content"])
                if audio:
                    st.audio(audio, format="audio/mp3")


# ---------- IMAGE GENERATION ----------
with st.expander("🎨 Generate Image (Replicate)", expanded=False):
    if not HAS_REPLICATE:
        st.warning("replicate quraşdırılmayıb. `pip install replicate` və REPLICATE_API_TOKEN env dəyişəni tələb olunur.")
    else:
        img_prompt = st.text_input("Image prompt:", key="img_prompt")
        if st.button("Generate") and img_prompt:
            with st.spinner("Şəkil yaradılır..."):
                try:
                    output = replicate.run(
                        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                        input={"prompt": img_prompt, "width": 768, "height": 768},
                    )
                    img_url = output[0] if isinstance(output, list) else output
                    st.image(img_url, caption=img_prompt)
                except Exception as e:
                    st.error(f"Xəta: {e}")


# ---------- FILE UPLOAD (per message) ----------
uploaded_file = st.file_uploader(
    "Attach file (optional):",
    type=["png", "jpg", "jpeg", "txt", "pdf"],
    key=f"uploader_{current_chat}",
)


# ---------- PREFILL FROM VOICE / QUICK PROMPT ----------
with st.expander("⚡ Quick Prompts", expanded=False):
    qcols = st.columns(4)
    quick = {
        "📝 Summarize": "Aşağıdakı mətni qısaca xülasə et: ",
        "🌐 Translate": "Bunu Azərbaycan dilinə tərcümə et: ",
        "💻 Code": "Bu funksiyanı Python-da yaz: ",
        "🔍 Explain": "Bu mövzunu ətraflı izah et: ",
    }
    for col, (label, template) in zip(qcols, quick.items()):
        if col.button(label, use_container_width=True):
            st.session_state.prefill = template
            st.rerun()


# ---------- CHAT INPUT ----------
initial_value = st.session_state.get("prefill", "")
prompt = st.chat_input("What would you like to ask Shadow?")

# Əgər prefill varsa və chat_input boşdursa, onu götür
if not prompt and initial_value:
    prompt = initial_value
    st.session_state.prefill = ""


if prompt:
    # File handling
    file_bytes = None
    file_type = None
    extracted_text = ""

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_type = uploaded_file.type

        if file_type == "text/plain":
            extracted_text = file_bytes.decode("utf-8", errors="ignore")
        elif file_type == "application/pdf":
            extracted_text = extract_pdf_text(file_bytes)
        elif file_type and file_type.startswith("image/") and HAS_OCR:
            extracted_text = ocr_image(file_bytes)

    # Save user message
    msg_data = {"role": "user", "content": prompt}
    if file_bytes and file_type and file_type.startswith("image/"):
        msg_data["image"] = file_bytes
    st.session_state.chats[current_chat].append(msg_data)

    with st.chat_message("user"):
        if msg_data.get("image"):
            st.image(msg_data["image"], width=300)
        st.markdown(prompt)

    # ---------- ASSISTANT ----------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("Shadow is thinking..."):
            # Web search
            search_data = web_search(prompt) if use_web_search else ""

            # RAG
            rag_data = query_vector_store(current_chat, prompt) if use_rag else ""

            # System prompt
            system_prompt = PERSONAS[persona_name]
            if search_data:
                system_prompt += f"\n\n[Real-time web search data]:\n{search_data}"
            if rag_data:
                system_prompt += f"\n\n[Relevant documents from vector store]:\n{rag_data}"
            if extracted_text:
                system_prompt += f"\n\n[File content provided by user]:\n{extracted_text[:6000]}"

            llm_messages = [{"role": "system", "content": system_prompt}]

            # Vision payload
            if file_bytes and file_type and file_type.startswith("image/"):
                b64 = base64.b64encode(file_bytes).decode("utf-8")
                llm_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{file_type};base64,{b64}"},
                        },
                    ],
                })
            else:
                llm_messages.append({"role": "user", "content": prompt})

            # Streaming
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
                full_response = f"❌ Xəta: {e}"
                placeholder.error(full_response)

        # Save assistant message
        st.session_state.chats[current_chat].append(
            {"role": "assistant", "content": full_response}
        )

        # Auto title
        if auto_title and len(st.session_state.chats[current_chat]) == 2:
            st.session_state.chats[current_chat][0]["title"] = generate_title(prompt, client)

    st.rerun()
