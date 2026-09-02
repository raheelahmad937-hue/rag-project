import subprocess

_real_run = subprocess.run
def _safe_run(*args, **kwargs):
    kwargs["check"] = False
    return _real_run(*args, **kwargs)
subprocess.run = _safe_run

from gpt4all import GPT4All

subprocess.run = _real_run

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# ---------- Page setup ----------
st.set_page_config(
    page_title="MLOps Q&A Assistant",
    page_icon="🤖",
    layout="centered",
)

# ---------- Custom styling ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        border-radius: 14px;
    }
    .stTextInput input {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🤖 MLOps Q&A Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions about your document — answers grounded in real retrieved content</p>', unsafe_allow_html=True)

# ---------- Load pipeline (cached so it only loads once) ----------
@st.cache_resource
def load_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path="./qdrant_data",
        collection_name="mlops_notes",
    )
    llm = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
    return vectorstore, llm

with st.spinner("Waking up the assistant... (first time takes a bit)"):
    vectorstore, llm = load_pipeline()

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Chat input ----------
question = st.chat_input("Ask something about your document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            results = vectorstore.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in results])

            prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

            with llm.chat_session():
                answer = llm.generate(prompt, max_tokens=300)

            st.markdown(answer)

            with st.expander("📄 Sources used"):
                for i, doc in enumerate(results, 1):
                    st.markdown(f"**Chunk {i}:** {doc.page_content[:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})