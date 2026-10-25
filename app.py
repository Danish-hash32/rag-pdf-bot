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
# Environment Setup (Groq API Key handling)
# ---------------------------------------------------------
GROQ_API_KEY = ""
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------
# Helper Functions: Process PDF, Summarize, & Generate Questions
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
            return None, None

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
        return vectorstore, docs

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        st.error(f"Error processing document: {e}")
        return None, None

def generate_summary_and_questions(docs, api_key):
    """Generates an executive summary and 3 suggested questions from initial pages."""
    if not docs or not api_key:
        return "", []
    
    sample_text = "\n\n".join([d.page_content for d in docs[:4]])[:3000]
    
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3
        )

        prompt = f"""
        Analyze the following document text and provide two things:
        1. A concise 2-3 sentence Executive Summary.
        2. Exactly 3 key questions a user might want to ask about this document.

        Format your output strictly like this:
        SUMMARY: <your summary here>
        QUESTIONS:
        - <Question 1>
        - <Question 2>
        - <Question 3>

        Document Text:
        {sample_text}
        """
        
        response = llm.invoke(prompt).content
        summary = ""
        questions = []
        
        if "SUMMARY:" in response:
            parts = response.split("QUESTIONS:")
            summary = parts[0].replace("SUMMARY:", "").strip()
            if len(parts) > 1:
                q_lines = parts[1].strip().split("\n")
                for line in q_lines:
                    cleaned_q = line.strip().lstrip("-*123. ").strip()
                    if cleaned_q:
                        questions.append(cleaned_q)
        return summary, questions[:3]
    except Exception:
        return "", []

# ---------------------------------------------------------
# Main App Workflow & Chat Logic
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    if "vectorstore" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
        with st.spinner("Indexing document & generating insights..."):
            vectorstore, raw_docs = process_pdf(uploaded_file)
            if vectorstore and raw_docs:
                st.session_state.vectorstore = vectorstore
                st.session_state.current_file = uploaded_file.name
                st.session_state.messages = []
                
                # Generate AI Summary & Suggested Questions
                summary, questions = generate_summary_and_questions(raw_docs, GROQ_API_KEY)
                st.session_state.doc_summary = summary
                st.session_state.suggested_questions = questions
                st.success("Document indexed successfully!")

# Display Executive Summary Box
if "doc_summary" in st.session_state and st.session_state.doc_summary:
    with st.expander("📊 **Document Executive Summary**", expanded=True):
        st.write(st.session_state.doc_summary)

# Display Suggested Quick Prompts
selected_question = None
if "suggested_questions" in st.session_state and st.session_state.suggested_questions:
    st.markdown("💡 **Suggested Questions:**")
    cols = st.columns(len(st.session_state.suggested_questions))
    for idx, q in enumerate(st.session_state.suggested_questions):
        if cols[idx].button(q, key=f"sq_{idx}"):
            selected_question = q

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process User Question
prompt = st.chat_input("Ask anything about your document...") or selected_question

if prompt:
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