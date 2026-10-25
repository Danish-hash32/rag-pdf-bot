import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from pypdf import PdfReader

# ---------------------------------------------------------
# Page Configuration & Header Banner
# ---------------------------------------------------------
st.set_page_config(page_title="DocuBrain AI", page_icon="🧠", layout="centered")

st.markdown("""
    <div style="background: linear-gradient(135deg, #0F172A, #1E293B); padding: 20px; border-radius: 12px; border: 1px solid #334155; text-align: center; margin-bottom: 20px;">
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
# Main Screen Reset Button
# ---------------------------------------------------------
col_reset1, col_reset2 = st.columns([3, 1])
with col_reset2:
    if st.button("🔄 Reset Chat", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ---------------------------------------------------------
# Sidebar Overview
# ---------------------------------------------------------
st.sidebar.title("📌 System Overview")
st.sidebar.markdown("👨‍💻 **Developer:** Muhammad Danish Haleem")
st.sidebar.markdown("🎓 BSAI Student | University of Agriculture Peshawar")
st.sidebar.markdown("---")
st.sidebar.info("Upload any PDF document to perform instant semantic query retrieval and AI analysis.")

# ---------------------------------------------------------
# Environment Setup (Groq API Key)
# ---------------------------------------------------------
GROQ_API_KEY = "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"

try:
    if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# ---------------------------------------------------------
# Crash-Free PDF Processing (In-Memory)
# ---------------------------------------------------------
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        reader = PdfReader(tmp_path)
        full_text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        if not full_text.strip():
            st.error("⚠️ Is PDF se text extract nahi ho saka.")
            return None

        from langchain_core.documents import Document
        docs = [Document(page_content=full_text)]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # In-Memory VectorStore (Zero Disk Writing = Zero Crashes)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)
        return vectorstore

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        st.error(f"Processing Error: {str(e)}")
        return None

# ---------------------------------------------------------
# Helper: Generate Document Insights
# ---------------------------------------------------------
def generate_insights(vectorstore):
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        sample_docs = retriever.invoke("summary overview introduction main points")
        sample_text = "\n\n".join([d.page_content for d in sample_docs[:3]])

        if not sample_text.strip():
            return "Document indexed successfully.", [
                "What is the main topic of this document?",
                "Can you summarize key points?",
                "What are the main conclusions?"
            ]

        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3
        )

        prompt = f"""Based on the following document text, provide:
1. A 2-line clean summary of the document.
2. Exactly 3 short questions a user can ask.

Format EXACTLY like this:
SUMMARY: <summary text>
QUESTIONS:
- <Question 1>
- <Question 2>
- <Question 3>

Text:
{sample_text}"""

        res = llm.invoke(prompt).content
        summary = ""
        questions = []

        if "SUMMARY:" in res:
            parts = res.split("QUESTIONS:")
            summary = parts[0].replace("SUMMARY:", "").strip()
            if len(parts) > 1:
                q_lines = parts[1].strip().split("\n")
                questions = [q.strip("- ").strip() for q in q_lines if q.strip()]

        return summary, questions[:3]
    except Exception:
        return "Document indexed successfully.", [
            "What is the main topic of this document?",
            "Can you summarize key points?",
            "What are the main conclusions?"
        ]

# ---------------------------------------------------------
# Main App Workflow
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], accept_multiple_files=False)

if uploaded_file:
    if "vectorstore" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
        with st.spinner("Indexing document & generating insights..."):
            vectorstore = process_pdf(uploaded_file)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.current_file = uploaded_file.name
                st.session_state.messages = []
                
                summary, suggested_qs = generate_insights(vectorstore)
                st.session_state.pdf_summary = summary
                st.session_state.suggested_questions = suggested_qs
                st.success("Document indexed successfully!")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Summary & Professional Suggested Queries
if "vectorstore" in st.session_state and st.session_state.vectorstore:
    if "pdf_summary" in st.session_state and st.session_state.pdf_summary:
        st.info(f"📌 **Document Summary:**\n{st.session_state.pdf_summary}")

    if "suggested_questions" in st.session_state and st.session_state.suggested_questions:
        st.markdown("💡 **Suggested Queries:**")
        q_cols = st.columns(len(st.session_state.suggested_questions))
        for idx, q_text in enumerate(st.session_state.suggested_questions):
            with q_cols[idx]:
                if st.button(f"📌 {q_text[:32]}...", key=f"sq_{idx}", use_container_width=True):
                    st.session_state.pending_prompt = q_text

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ONLY show Chat Input if PDF is successfully indexed
if "vectorstore" in st.session_state and st.session_state.vectorstore:
    user_input = st.chat_input("Ask anything about your document...")
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pending_prompt
        del st.session_state.pending_prompt
    else:
        prompt = user_input

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

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