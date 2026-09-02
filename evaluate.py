import subprocess

_real_run = subprocess.run
def _safe_run(*args, **kwargs):
    kwargs["check"] = False
    return _real_run(*args, **kwargs)
subprocess.run = _safe_run

from gpt4all import GPT4All

subprocess.run = _real_run

import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import numpy as np

# ---------- Our answer key (test set) ----------
TEST_SET = [
    {
        "question": "What is MLOps?",
        "expected_answer": "MLOps combines machine learning, software engineering, automation, deployment, monitoring, and data management practices to make ML systems reliable, repeatable, maintainable, and scalable in production.",
        "keywords": ["software engineering", "automation", "deployment", "monitoring"],
    },
    {
        "question": "What are the stages of the machine learning lifecycle?",
        "expected_answer": "Data collection, data preparation, model training, model evaluation, deployment, monitoring, and retraining.",
        "keywords": ["data collection", "model training", "deployment", "monitoring"],
    },
    {
        "question": "What is data drift?",
        "expected_answer": "Data drift happens when the data used in production changes over time, so a model trained on old behavior may not match new behavior.",
        "keywords": ["data drift", "production", "change"],
    },
    {
        "question": "What is overfitting?",
        "expected_answer": "Overfitting happens when a model learns the training data too closely, including noise and unusual patterns, and performs poorly on new data.",
        "keywords": ["overfitting", "training data", "noise"],
    },
    {
        "question": "What is Docker used for in machine learning?",
        "expected_answer": "Docker packages the application code, Python version, dependencies, and configuration so development and production environments match.",
        "keywords": ["docker", "container", "dependencies"],
    },
    {
        "question": "What does a RAG system do with documents?",
        "expected_answer": "A RAG system collects documents, splits them into chunks, converts them into embeddings, and stores them in a vector database, then retrieves relevant chunks to help an LLM answer questions.",
        "keywords": ["chunks", "embeddings", "vector database", "retrieves"],
    },
]

# ---------- Load pipeline ----------
print("Loading pipeline...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    path="./qdrant_data",
    collection_name="mlops_notes",
)
llm = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")


def cosine_similarity(vec_a, vec_b):
    vec_a, vec_b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def score_retrieval(retrieved_chunks, keywords):
    """What % of expected keywords showed up in the retrieved chunks?"""
    combined_text = " ".join([c.page_content.lower() for c in retrieved_chunks])
    found = sum(1 for kw in keywords if kw.lower() in combined_text)
    return found / len(keywords)


# ---------- Run evaluation ----------
results = []

for i, item in enumerate(TEST_SET, 1):
    print(f"\n[{i}/{len(TEST_SET)}] Asking: {item['question']}")

    # Retrieve
    retrieved = vectorstore.similarity_search(item["question"], k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved])

    # Generate answer
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {item['question']}

Answer:"""
    with llm.chat_session():
        answer = llm.generate(prompt, max_tokens=300)

    # Score retrieval
    retrieval_score = score_retrieval(retrieved, item["keywords"])

    # Score answer similarity
    answer_emb = embeddings.embed_query(answer)
    expected_emb = embeddings.embed_query(item["expected_answer"])
    answer_score = cosine_similarity(answer_emb, expected_emb)

    results.append({
        "question": item["question"],
        "generated_answer": answer,
        "expected_answer": item["expected_answer"],
        "retrieval_score": round(retrieval_score, 2),
        "answer_similarity_score": round(answer_score, 2),
    })

    print(f"  Retrieval score: {retrieval_score:.2f} | Answer similarity: {answer_score:.2f}")

# ---------- Report ----------
avg_retrieval = sum(r["retrieval_score"] for r in results) / len(results)
avg_answer = sum(r["answer_similarity_score"] for r in results) / len(results)

print("\n" + "=" * 50)
print("EVALUATION REPORT")
print("=" * 50)
print(f"Average retrieval score:        {avg_retrieval:.2f} / 1.00")
print(f"Average answer similarity:      {avg_answer:.2f} / 1.00")
print("=" * 50)

with open("eval_results.json", "w") as f:
    json.dump({
        "average_retrieval_score": round(avg_retrieval, 2),
        "average_answer_similarity_score": round(avg_answer, 2),
        "details": results,
    }, f, indent=2)

print("\nFull results saved to eval_results.json")