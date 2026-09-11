import streamlit as st
from groq import Groq
from ddgs import DDGS
import uuid
import base64

st.set_page_config(page_title="Shadow AI - Advanced", page_icon="🤖", layout="centered")

# Sol panel (Sidebar) - Söhbət tarixçəsi və tənzimləmələr
with st.sidebar:
    st.header("⚙️ Settings")
    user_api_key = st.text_input("Enter your Groq API Key:", type="password")
    st.markdown("---")
    
    st.subheader("💬 Chats")
    
    # Söhbətləri yadda saxlamaq üçün session_state
    if "chats" not in st.session_state:
        st.session_state.chats = {} 
    if "current_chat_id" not in st.session_state:
        first_id = str(uuid.uuid4())[:8]
        st.session_state.chats[first_id] = []
        st.session_state.current_chat_id = first_id

    # Yeni söhbət yaratma düyməsi
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.chats[new_id] = []
        st.session_state.current_chat_id = new_id
        st.rerun()

    # Cari söhbəti təmizləmək/silmək üçün düymə
    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        current_id = st.session_state.current_chat_id
        if current_id in st.session_state.chats:
            st.session_state.chats[current_id] = []
        st.rerun()

    st.markdown("---")
    st.markdown("### History")
    
    # Mövcud söhbətlərin siyahısı
    for chat_id in list(st.session_state.chats.keys()):
        messages_in_chat = st.session_state.chats[chat_id]
        if messages_in_chat:
            chat_title = messages_in_chat[0]["content"][:20] + "..."
        else:
            chat_title = f"Chat {chat_id}"
            
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(chat_title, key=f"chat_btn_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with col2:
            # Söhbətin özünü tamamilə silmək üçün düymə (əgər birdən çox chat varsa)
            if len(st.session_state.chats) > 1:
                if st.button("❌", key=f"del_btn_{chat_id}"):
                    del st.session_state.chats[chat_id]
                    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                    st.rerun()

    st.markdown("---")
    st.markdown("💡 *Get a free Groq API key from [Groq Console](https://console.groq.com/).*")

# Əsas səhifə başlığı
st.title("🤖 Shadow - Smart Assistant")
st.caption("An AI assistant with internet access, file upload, and chat management")

# Veb axtarış funksiyası
def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
        return " ".join(results)
    except Exception as e:
        return ""

# Cari söhbətin mesajlarını yükləyək
current_chat = st.session_state.current_chat_id
if current_chat not in st.session_state.chats:
    st.session_state.chats[current_chat] = []

# Köhnə mesajları ekranda göstər
for message in st.session_state.chats[current_chat]:
    with st.chat_message(message["role"]):
        if "image" in message and message["image"]:
            st.image(message["image"], width=300)
        st.markdown(message["content"])

# Fayl/Şəkil yükləmək üçün widget
uploaded_file = st.file_uploader("Upload an image or file (optional):", type=["png", "jpg", "jpeg", "txt", "pdf"])

# İstifadəçidən mesaj alaq
if prompt := st.chat_input("What would you like to ask Shadow?"):
    if not user_api_key:
        st.error("Please enter your Groq API key from the sidebar first!")
        st.stop()
        
    file_bytes = None
    file_base64 = None
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")

    # Mesajı tarixçəyə əlavə edirik
    msg_data = {"role": "user", "content": prompt}
    if uploaded_file is not None and uploaded_file.type.startswith("image/"):
        msg_data["image"] = file_bytes

    st.session_state.chats[current_chat].append(msg_data)
    
    with st.chat_message("user"):
        if uploaded_file is not None and uploaded_file.type.startswith("image/"):
            st.image(file_bytes, width=300)
        st.markdown(prompt)

    # Assistentin cavabı
    with st.chat_message("assistant"):
        with st.spinner("Shadow is thinking and searching the web..."):
            search_data = web_search(prompt)
            
            system_prompt = f"""
            Your name is Shadow. You are a high-level analyst, an AI assistant with internet access and deep reasoning capabilities.
            Important rule: Always respond in the exact language the user is writing in.
            Use the following real-time web search data when answering the user's question:
            
            [Web Data]:
            {search_data}
            """

            try:
                client = Groq(api_key=user_api_key)
                
                # Əgər şəkil yüklənibsə və model dəstəkləyirsə vizual məlumatı da nəzərə alaq
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model="openai/gpt-oss-120b",
                    temperature=0.5,
                )
                response = chat_completion.choices[0].message.content
            except Exception as e:
                response = f"An error occurred: {str(e)}"
            
            st.markdown(response)
            st.session_state.chats[current_chat].append({"role": "assistant", "content": response})
