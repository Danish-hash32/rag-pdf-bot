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
st.set_page_config(page_title="PDF RAG Bot", page_icon="💬", layout="centered")

st.title("💬 Chat with your PDF")

# Header Developer Banner
st.markdown("""
    <div style="background-color:#1E293B; padding:10px; border-radius:8px; text-align:center; margin-bottom:15px;">
        <p style="color:#38BDF8; font-size:16px; font-weight:bold; margin:0;">
            🚀 Developed by Muhammad Danish Haleem
        </p>
        <p style="color:#94A3B8; font-size:12px; margin:0;">
            BSAI Student | University of Agriculture Peshawar
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Navigation & Reset
st.sidebar.title("📌 Navigation")
st.sidebar.markdown("👨‍💻 **Developer:** Muhammad Danish Haleem")
st.sidebar.markdown("🎓 BSAI Student | UOA Peshawar")
st.sidebar.markdown("---")

if st.sidebar.button("🧹 Clear Chat & Reset Memory"):
    st.session_state.clear()
    st.rerun()

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
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.remove(tmp_path)

        if not docs:
            st.error("⚠️ Is PDF se koi text read nahi ho saka. Baraye meharbani check karein ke PDF scanned image toh nahi hai.")
            return None

        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Create Embeddings & In-Memory Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if "cryptography" in str(e).lower() or "dependencyerror" in str(e).lower():
            st.error("🔒 Yeh PDF encrypted/password-protected hai. System ko `cryptography` package ki zaroorat hai.")
        else:
            st.error(f"⚠️ PDF Process karne mein masla aaya: {e}")
        return None

# ---------------------------------------------------------
# PDF Uploading & State Management
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF file to begin", type=["pdf"])

if uploaded_file:
    # Auto-Reset if a new file is uploaded
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.chat_history = []
        with st.spinner("PDF process ho rahi hai, baraye meharbani intazar karein..."):
            st.session_state.retriever = process_pdf(uploaded_file)

    if st.session_state.get("retriever"):
        st.success("✅ PDF successfully load ho gayi! Ab aap sawal pooch sakte hain.")

        # Initialize Chat History
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display Chat Messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat Input Box
        user_query = st.chat_input("Ask a question about your uploaded PDF...")

        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            if not GROQ_API_KEY:
                st.error("Groq API Key nahi mili. Secrets / Environment variables check karein.")
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Jawab dhundha ja raha hai..."):
                        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                        
                        # Fetch relevant context documents
                        docs = st.session_state.retriever.invoke(user_query)
                        context_text = "\n\n".join([doc.page_content for doc in docs])

                        prompt = f"""
                        Answer the user's question based ONLY on the provided context below.
                        If the answer is not contained within the context, politely state that the information is not present in the document.

                        Context:
                        {context_text}

                        Question:
                        {user_query}
                        """

                        response = llm.invoke(prompt)
                        answer = response.content
                        
                        st.write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})