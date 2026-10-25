import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# ---------------------------------------------------------
# Page Configuration & Header
# ---------------------------------------------------------
st.set_page_config(page_title="DocuBrain AI", page_icon="🧠", layout="centered")

# Display Logo Image (Top Banner)
st.image("logo.png.jpeg", use_container_width=True)

# Sub-header Banner with Developer Info
st.markdown("""
    <div style="background: linear-gradient(135deg, #0F172A, #1E293B); padding: 18px; border-radius: 12px; border: 1px solid #334155; text-align: center; margin-bottom: 20px;">
        <h2 style="color: #F8FAFC; margin: 0; font-family: sans-serif;">🧠 DocuBrain AI</h2>
        <p style="color: #38BDF8; font-size: 13px; margin: 4px 0 0 0; font-weight: 500;">
            Enterprise Document Intelligence Workspace
        </p>
        <p style="color: #94A3B8; font-size: 12px; margin: 6px 0 0 0; font-weight: 600;">
            🚀 Developed by Muhammad Danish Haleem | BSAI Student, UOA Peshawar
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
# Environment Setup (Groq API Key)
# ---------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------
# Helper Function: Process PDF (Fresh Unique Vectorstore)
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

        # Unique Collection Name to avoid memory leakage between different PDFs
        collection_name = f"doc_{hash(uploaded_file.name) & 0xFFFFFFFF}"

        # Vector Store Creation
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
# Main App Workflow & Chat Logic
# ---------------------------------------------------------
if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY is missing. Please set it in Streamlit Secrets or Environment Variables.")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    if "vectorstore" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
        with st.spinner("Indexing document vectors..."):
            vectorstore = process_pdf(uploaded_file)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.current_file = uploaded_file.name
                st.session_state.messages = []
                st.success("Document indexed successfully!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process User Question
if prompt := st.chat_input("Ask anything about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" in st.session_state and st.session_state.vectorstore:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
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
    else:
        with st.chat_message("assistant"):
            st.markdown("Please upload a PDF document first to begin answering queries.")