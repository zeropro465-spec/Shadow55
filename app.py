import streamlit as st
from groq import Groq
from ddgs import DDGS
import uuid

st.set_page_config(page_title="Shadow AI - Chat History", page_icon="🤖", layout="centered")

# Sol panel (Sidebar) - Söhbət tarixçəsi və tənzimləmələr
with st.sidebar:
    st.header("⚙️ Settings")
    user_api_key = st.text_input("Enter your Groq API Key:", type="password")
    st.markdown("---")
    
    st.subheader("💬 Chats")
    
    # Söhbətləri yadda saxlamaq üçün session_state
    if "chats" not in st.session_state:
        st.session_state.chats = {} # {chat_id: [messages]}
    if "current_chat_id" not in st.session_state:
        # İlkin olaraq boş bir söhbət yaradaq
        first_id = str(uuid.uuid4())[:8]
        st.session_state.chats[first_id] = []
        st.session_state.current_chat_id = first_id

    # Yeni söhbət yaratma düyməsi
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.chats[new_id] = []
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.markdown("---")
    st.markdown("### History")
    
    # Mövcud söhbətlərin siyahısı (düymə şəklində)
    for chat_id in list(st.session_state.chats.keys()):
        # Söhbətin ilk mesajından başlıq düzəldək və ya sadəcə ID göstərək
        messages_in_chat = st.session_state.chats[chat_id]
        if messages_in_chat:
            chat_title = messages_in_chat[0]["content"][:20] + "..."
        else:
            chat_title = f"Chat {chat_id}"
            
        if st.button(chat_title, key=f"chat_btn_{chat_id}", use_container_width=True):
            st.session_state.current_chat_id = chat_id
            st.rerun()

    st.markdown("---")
    st.markdown("💡 *Get a free Groq API key from [Groq Console](https://console.groq.com/).*")

# Əsas səhifə başlığı
st.title("🤖 Shadow - Smart Assistant")
st.caption("An AI assistant with internet access and deep reasoning capabilities")

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
        st.markdown(message["content"])

# İstifadəçidən mesaj alaq
if prompt := st.chat_input("What would you like to ask Shadow?"):
    if not user_api_key:
        st.error("Please enter your Groq API key from the sidebar first!")
        st.stop()
        
    # İstifadəçi mesajını tarixçəyə əlavə edib göstəririk
    st.session_state.chats[current_chat].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistentin cavabı
    with st.chat_message("assistant"):
        with st.spinner("Shadow is thinking and searching the web..."):
            search_data = web_search(prompt)
            
            system_prompt = f"""
            Your name is Shadow. You are a high-level analyst, an AI assistant with internet access and deep reasoning capabilities.
            Important rule: Always respond in the exact language the user is writing in.
            When performing complex tasks, think step-by-step first (Chain of Thought).
            Use the following real-time web search data when answering the user's question:
            
            [Web Data]:
            {search_data}
            """

            try:
                client = Groq(api_key=user_api_key)
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
