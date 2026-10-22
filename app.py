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
st.set_page_config(page_title="PDF AI Assistant", page_icon="📄", layout="centered")

st.title("📄 Enterprise PDF AI Assistant")

# Header Developer Banner
st.markdown("""
    <div style="background-color:#1E293B; padding:12px; border-radius:8px; text-align:center; margin-bottom:20px;">
        <p style="color:#38BDF8; font-size:16px; font-weight:bold; margin:0;">
            🚀 Developed by Muhammad Danish Haleem
        </p>
        <p style="color:#94A3B8; font-size:13px; margin:0;">
            BSAI Student | University of Agriculture Peshawar
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Environment Setup (Groq API Key)
# ---------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------
# Helper Function: Process PDF (In-Memory Vectorstore)
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
            st.error("⚠️ Unable to extract text from this PDF. Please ensure the document is not an image-only scan.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if "cryptography" in str(e).lower() or "dependencyerror" in str(e).lower():
            st.error("🔒 The uploaded PDF is encrypted or password-protected.")
        else:
            st.error(f"⚠️ Error processing PDF: {str(e)}")
        return None

# ---------------------------------------------------------
# Main Application UI & Logic
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF document to begin", type=["pdf"])

# Clear / Reset Button directly on Main Screen
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Reset Chat", use_container_width=True):
        st.session_state.clear()
        st.rerun()

if uploaded_file:
    # Auto-Reset state when a new file is uploaded
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.chat_history = []
        with st.spinner("Processing PDF document... Please wait."):
            st.session_state.retriever = process_pdf(uploaded_file)

    if st.session_state.get("retriever"):
        st.success("✅ Document processed successfully. You can now ask questions below.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Render Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat Input Box
        user_query = st.chat_input("Ask a question about your uploaded document...")

        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            if not GROQ_API_KEY:
                st.error("Groq API Key is missing. Please verify your Streamlit Secrets.")
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing document and generating response..."):
                        # Updated to current stable active model
                        llm = ChatGroq(
                            groq_api_key=GROQ_API_KEY, 
                            model_name="llama-3.1-8b-instant"
                        )
                        
                        docs = st.session_state.retriever.invoke(user_query)
                        context_text = "\n\n".join([doc.page_content for doc in docs])

                        prompt = f"""
                        You are a professional AI Assistant. Answer the user's question based ONLY on the provided context below.
                        If the answer cannot be found in the context, state clearly that the information is not available in the uploaded document.

                        Context:
                        {context_text}

                        Question:
                        {user_query}
                        """

                        try:
                            response = llm.invoke(prompt)
                            answer = response.content
                            st.write(answer)
                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        except Exception as err:
                            st.error(f"Error querying LLM API: {str(err)}")