import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

st.set_page_config(page_title="PDF RAG Bot", page_icon="🤖")
st.title("🤖 Chat with your PDF")

# 1. Groq Setup
# Local test ke liye yahan apni key rakhein (Cloud par daalte waqt Secrets use karenge)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "") 
client = Groq(api_key=GROQ_API_KEY)

# Dynamic Active Chat Model Filter
@st.cache_resource
def get_active_chat_model():
    try:
        models = client.models.list()
        ignore_keywords = ["guard", "embed", "whisper", "audio", "canopy", "orpheus", "safeguard", "vision", "prompt"]
        valid_models = []
        for m in models.data:
            mid = m.id.lower()
            if not any(k in mid for k in ignore_keywords):
                valid_models.append(m.id)
        if valid_models:
            return valid_models[0]
        return models.data[0].id
    except Exception:
        return "llama-3.1-8b-instant"

active_model_id = get_active_chat_model()

# 2. File Upload UI
uploaded_file = st.file_uploader("Upload a PDF file to begin", type=["pdf"])

@st.cache_resource
def process_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    os.remove(tmp_path)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

if uploaded_file:
    retriever = process_pdf(uploaded_file.getvalue())
    st.success("PDF Loaded Successfully! Ask your questions below.")

    # 3. Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your uploaded PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating answer..."):
                relevant_docs = retriever.invoke(prompt)
                
                raw_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                context_text = raw_text[:1000]
                
                rag_prompt = (
                    f"You are a helpful AI assistant. Answer the user's question clearly using ONLY the following PDF content:\n\n"
                    f"{context_text}\n\n"
                    f"Question: {prompt}\n\n"
                    f"Answer:"
                )
                
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": rag_prompt}],
                    model=active_model_id,
                    max_tokens=300
                )
                
                ans = chat_completion.choices[0].message.content
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
else:
    st.info("Please upload a PDF file to start chatting.")