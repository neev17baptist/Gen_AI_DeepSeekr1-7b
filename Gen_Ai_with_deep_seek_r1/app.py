import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    /* Global Styling */
    html, body, .main {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(to right, #f8f9fd, #e3f2fd);
        color: #2c3e50;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #007bff, #00c6ff);
        color: white;
        border-radius: 10px;
    }
    
    /* Input Fields */
    .stTextInput textarea {
        background: #ffffff;
        color: #2c3e50;
        border: 2px solid #007bff;
        border-radius: 8px;
        padding: 8px;
    }
    
    /* Selectbox Styling */
    .stSelectbox div[data-baseweb="select"] {
        background: #ffffff;
        color: #2c3e50;
        border: 2px solid #007bff;
        border-radius: 8px;
    }
    
    /* Dropdown Menu */
    div[role="listbox"] div {
        background: #f8f9fd;
        color: #2c3e50;
        border-bottom: 1px solid #ddd;
    }
    
    /* Headings */
    h1, h2, h3 {
        font-weight: 600;
        color: #0056b3;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(to right, #007bff, #00c6ff);
        color: white;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:7b", "deepseek-r1:3b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")


# initiate the chat engine

llm_engine=ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    processing_pipeline=prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})

    # Rerun to update chat display
    st.rerun()