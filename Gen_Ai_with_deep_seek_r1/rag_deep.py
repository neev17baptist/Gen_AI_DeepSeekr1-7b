import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    /* Global App Styling */
    
    .stApp {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(to right, #e3f2fd, #ffffff);
        color: #2c3e50;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background: white;
        color: #2c3e50;
        border: 2px solid #007bff;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background: rgba(0, 123, 255, 0.1);
        border: 2px solid #007bff;
        color: #2c3e50;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background: rgba(0, 198, 255, 0.1);
        border: 2px solid #00c6ff;
        color: #2c3e50;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background: linear-gradient(to right, #007bff, #00c6ff);
        color: white;
        font-weight: bold;
    }
    
    /* File Uploader */
    .stFileUploader {
        background: #ffffff;
        border: 2px solid #007bff;
        border-radius: 8px;
        padding: 15px;
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



PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""
PDF_STORAGE_PATH = 'C:/Users/neevb/OneDrive/Desktop/'
# "C:\Users\neevb\OneDrive\Desktop\Prog_fy.pdf"
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:7b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:7b")


def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})


# UI Configuration


st.title("📘 DocuMind AI, by neXus")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False

)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)