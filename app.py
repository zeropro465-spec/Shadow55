# app.py
import streamlit as st
from groq import Groq
from ddgs import DDGS

st.set_page_config(page_title="Shadow AI", page_icon="🤖", layout="centered")

st.title("🤖 Shadow - Smart Assistant")
st.caption("An AI assistant with internet access and deep reasoning capabilities")

with st.sidebar:
    st.header("Settings")
    user_api_key = st.text_input("Enter your Groq API Key:", type="password")
    st.markdown("---")
    st.markdown("💡 *Don't have an API key? You can get a free one from the [Groq Console](https://console.groq.com/).*")

def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
        return " ".join(results)
    except Exception as e:
        return ""

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to ask Shadow?"):
    if not user_api_key:
        st.error("Please enter your Groq API key from the sidebar first!")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

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
            st.session_state.messages.append({"role": "assistant", "content": response})