import os
from pathlib import Path

import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
 

# ------------------- Streamlit Setup -------------------
st.set_page_config(page_title="RAG Chatbot - Gemini 2.5 Flash", layout="wide")
st.title("Smart Document Intelligence Assistant for GS Office Records")



# ---- All Text Primary Color ----
st.markdown(
    """
    <style>
    body, .stApp, .stApp * {
        color: #2c5530 !important;
    }
    /* File uploader button: yellow background, white text */
    .stApp .stFileUploader > div > button {
        background-color: #ffd700 !important;
        color: #fff !important;
        border: 1.5px solid #ffd700 !important;
        border-radius: 8px !important;
        font-weight: 500;
    }
    .stApp .stFileUploader > div > button:hover {
        background-color: #ffed4e !important;
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
        """
Upload PDF files and ask questions about their content.  
The bot will extract, summarize and chat to answer your queries.
"""
)

# ------------------- Google API Key -------------------
def load_google_api_key():
    # 1) Check Streamlit secrets (deployed on Streamlit Cloud)
    if hasattr(st, 'secrets') and st.secrets.get('GOOGLE_API_KEY'):
        return st.secrets.get('GOOGLE_API_KEY')

    # 2) Check environment variables
    if os.environ.get('GOOGLE_API_KEY'):
        return os.environ.get('GOOGLE_API_KEY')

    # 3) Fallback to a local .env file during development (not committed)
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        if os.environ.get('GOOGLE_API_KEY'):
            return os.environ.get('GOOGLE_API_KEY')

    return None


GOOGLE_API_KEY = load_google_api_key()
if not GOOGLE_API_KEY:
    st.warning("⚠️ Google API key not found. Provide it via Streamlit secrets, the environment variable `GOOGLE_API_KEY`, or a local `.env` file.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ------------------- Initialize Models -------------------
@st.cache_resource(show_spinner=False)
def load_models():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    return embeddings, chat_model

embeddings, chat_model = load_models()

# ------------------- Vector Store -------------------
PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(PERSIST_DIR, exist_ok=True)
vectordb = None

# ------------------- PDF Processing -------------------
def load_pdf(file) -> str:
    """Extract text from PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def process_documents(documents):
    """Split text into chunks and create Chroma vector store."""
    global vectordb
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()

# ------------------- Query Function -------------------
def get_answer(query: str) -> str:
    if vectordb:
        retriever = vectordb.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=retriever
        )
        return qa.run(query)
    else:
        return "Please upload a PDF document first."

# ------------------- Streamlit UI -------------------
uploaded_files = st.file_uploader(" Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    progress = st.progress(0)
    for i, uploaded_file in enumerate(uploaded_files):
        pdf_text = load_pdf(uploaded_file)
        if pdf_text.strip():
            doc = Document(page_content=pdf_text, metadata={"source": uploaded_file.name})
            all_docs.append(doc)
        progress.progress(int((i+1)/len(uploaded_files)*100))
    process_documents(all_docs)
    st.success(f"✅ Processed {len(all_docs)} PDF(s) into vector store!")

query = st.text_input(" Ask a question about your PDFs:")
if query:
    with st.spinner("🤖 Thinking..."):
        response = get_answer(query)
        st.write(response)

# ------------------- Footer -------------------
st.markdown("---")
st.markdown(
    """
**Notes:**
- Ensure your Google API key has access to the Generative Language API.
- Vector store is saved locally in `chroma_db/`.
- For large PDFs, processing may take a few seconds.
"""
)
