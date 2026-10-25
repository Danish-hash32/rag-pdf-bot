import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# ---------------------------------------------------------
# Page Configuration & Clean Header Banner
# ---------------------------------------------------------
st.set_page_config(page_title="DocuBrain AI", page_icon="🧠", layout="centered")

st.markdown("""
    <div style="background: linear-gradient(135deg, #0F172A, #1E293B); padding: 20px; border-radius: 12px; border: 1px solid #334155; text-align: center; margin-bottom: 25px;">
        <h2 style="color: #F8FAFC; margin: 0; font-family: sans-serif; font-weight: 700;">🧠 DocuBrain AI</h2>
        <p style="color: #38BDF8; font-size: 14px; margin: 6px 0 0 0; font-weight: 600;">
            Enterprise Document Intelligence Workspace
        </p>
        <p style="color: #94A3B8; font-size: 12px; margin: 8px 0 0 0; font-weight: 500;">
            🚀 Developed by <b>Muhammad Danish Haleem</b> | BSAI Student, UOA Peshawar
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("📌 System Overview")
st.sidebar.markdown("👨‍💻 **Developer:** Muhammad Danish Haleem")
st.sidebar.markdown("🎓 BSAI Student | University of Agriculture Peshawar")
st.sidebar.markdown("---")
st.sidebar.info("Upload any PDF document to perform instant semantic query retrieval and AI analysis.")

# ---------------------------------------------------------
# Environment Setup (Groq API Key Safe Handling)
# ---------------------------------------------------------
GROQ_API_KEY = ""
try:
    if "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not GROQ_API_KEY:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------
# Helper Functions: Process PDF
# ---------------------------------------------------------
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        if not docs:
            st.error("⚠️ Unable to extract text. Please verify that the PDF is not an image-only scan.")
            return None

        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Embedding Setup
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        collection_name = f"doc_{hash(uploaded_file.name) & 0xFFFFFFFF}"

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name
        )
        return vectorstore

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        st.error(f"Error processing document: {e}")
        return None

# ---------------------------------------------------------
# Main App Workflow & UI Elements
# ---------------------------------------------------------
if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY missing! Streamlit Cloud Settings > Secrets mein GROQ_API_KEY add karein.")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], accept_multiple_files=False)

if uploaded_file:
    if "vectorstore" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
        with st.spinner("Indexing document..."):
            vectorstore = process_pdf(uploaded_file)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.current_file = uploaded_file.name
                st.session_state.messages = []
                st.success("Document indexed successfully!")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ONLY show Chat Input if a valid vectorstore exists (PDF successfully loaded)
if "vectorstore" in st.session_state and st.session_state.vectorstore:
    if prompt := st.chat_input("Ask anything about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not GROQ_API_KEY:
            with st.chat_message("assistant"):
                st.error("GROQ_API_KEY is not configured. Please add it to secrets.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                        relevant_docs = retriever.invoke(prompt)
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])

                        llm = ChatGroq(
                            groq_api_key=GROQ_API_KEY,
                            model_name="llama-3.3-70b-versatile",
                            temperature=0.2
                        )

                        system_prompt = f"Use the following context to answer the user's question accurately.\n\nContext:\n{context}"
                        messages = [
                            ("system", system_prompt),
                            ("user", prompt)
                        ]

                        response = llm.invoke(messages)
                        st.markdown(response.content)
                        st.session_state.messages.append({"role": "assistant", "content": response.content})
                    except Exception as err:
                        st.error(f"Failed to generate response: {err}")