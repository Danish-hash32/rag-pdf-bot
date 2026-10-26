import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# ---------------------------------------------------------
# Page Configuration & Header
# ---------------------------------------------------------
st.set_page_config(page_title="DocuMind AI", page_icon="⚡", layout="centered")

st.markdown("""
    <div style="background: linear-gradient(135deg, #0F172A, #1E293B); padding: 18px; border-radius: 12px; border: 1px solid #334155; text-align: center; margin-bottom: 20px;">
        <h2 style="color: #F8FAFC; margin: 0; font-family: sans-serif;">⚡ DocuMind AI</h2>
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
st.sidebar.info("Running locally via Hugging Face. No API keys required!")

# ---------------------------------------------------------
# Load Local Hugging Face LLM (Optimized for Speed & No Loops)
# ---------------------------------------------------------
@st.cache_resource
def load_local_llm():
    model_id = "google/gemma-2-2b-it"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype="auto"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=90,         # Shorter length for faster response time
            temperature=0.01,          # Precise answering
            do_sample=False,           # Greedy decoding to avoid loops
            repetition_penalty=1.2     # Prevents repeating sentences
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception:
        model_id = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=90)
        return HuggingFacePipeline(pipeline=pipe)

with st.spinner("Loading local AI model (Takes a moment on first load)..."):
    llm = load_local_llm()

# ---------------------------------------------------------
# Helper Function: Process PDF
# ---------------------------------------------------------
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        if not docs:
            st.error("⚠️ Unable to extract text. Please verify that the PDF is not an image-only scan.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        collection_name = f"pdf_{hash(uploaded_file.name)}"
        
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            collection_name=collection_name
        )
        
        return vectorstore.as_retriever(search_kwargs={"k": 2})

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        st.error(f"⚠️ Error processing PDF document: {str(e)}")
        return None

# ---------------------------------------------------------
# Main Interface & Controls
# ---------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

with col2:
    st.write("") 
    st.write("")
    if st.button("🔄 Reset Chat", use_container_width=True):
        st.session_state.clear()
        st.rerun()

if uploaded_file:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.chat_history = []
        if "retriever" in st.session_state:
            del st.session_state["retriever"]
        
        with st.spinner("Analyzing and indexing document structure..."):
            st.session_state.retriever = process_pdf(uploaded_file)

    if st.session_state.get("retriever"):
        st.success("✅ Document processed and indexed. You may ask questions below.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_query = st.chat_input("Ask a question regarding the uploaded PDF...")

        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Searching document context and generating answer..."):
                    try:
                        docs = st.session_state.retriever.invoke(user_query)
                        context_text = "\n\n".join([doc.page_content for doc in docs])

                        formatted_prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\nAnswer:"

                        response = llm.invoke(formatted_prompt)
                        
                        if "Answer:" in response:
                            answer = response.split("Answer:")[-1].strip()
                        else:
                            answer = response.strip()
                        
                        st.write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as err:
                        st.error(f"Error generating response: {str(err)}")