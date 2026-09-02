import subprocess

# Workaround for a gpt4all bug on Intel Macs (sysctl.proc_translated check crashes)
_real_run = subprocess.run
def _safe_run(*args, **kwargs):
    kwargs["check"] = False
    return _real_run(*args, **kwargs)
subprocess.run = _safe_run

from gpt4all import GPT4All

subprocess.run = _real_run  # put it back to normal right after

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from gpt4all import GPT4All

# 1. Reconnect to our vector database (the librarian we already built)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    path="./qdrant_data",
    collection_name="mlops_notes",
)

# 2. Load a small free local LLM (downloads automatically on first run, ~4GB)
llm = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

# 3. Ask a question in a loop
while True:
    question = input("\nAsk a question (or type 'quit'): ")
    if question.lower() == "quit":
        break

    # Find the most relevant chunks (the "similarity search" step)
    results = vectorstore.similarity_search(question, k=3)  # top 3 matching chunks

    # Combine those chunks into one block of context
    context = "\n\n".join([doc.page_content for doc in results])

    # Build a prompt that gives the LLM the context + question
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}


Question: {question}

Answer:"""

    # 4. Get the answer from the LLM
    with llm.chat_session():
        answer = llm.generate(prompt, max_tokens=300)

    print(f"\nAnswer: {answer}")