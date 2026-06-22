import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_STORE_PATH = "vector_store"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

print("Starting RAG pipeline...")

# ============================================================
# LOAD VECTOR STORE FILES
# ============================================================

faiss_index_path = os.path.join(VECTOR_STORE_PATH, "faiss_index.bin")
chunks_path = os.path.join(VECTOR_STORE_PATH, "chunks.pkl")
metadata_path = os.path.join(VECTOR_STORE_PATH, "metadata.pkl")

index = faiss.read_index(faiss_index_path)

with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

print("Vector store loaded successfully")

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded")

# ============================================================
# RETRIEVER FUNCTION
# ============================================================

def retrieve(query, top_k=TOP_K):
    query_embedding = model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        if i == -1:
            continue
        if i < len(chunks):
            results.append({
                "text": chunks[i],
                "metadata": metadata[i]
            })

    return results

# ============================================================
# PROMPT TEMPLATE
# ============================================================

def build_prompt(context, question):
    return f"""
You are a financial analyst assistant for CrediTrust.

Your task is to answer questions about customer complaints.

Use ONLY the context provided below.
If the answer is not available in the context, say "I don't have enough information".

Context:
{context}

Question:
{question}

Answer:
""".strip()

# ============================================================
# LLM (FLAN-T5)
# ============================================================

print("Loading LLM...")
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)
print("LLM loaded")

def generate_answer(prompt):
    response = generator(prompt)
    return response[0]["generated_text"]

# ============================================================
# RAG PIPELINE FUNCTION
# ============================================================

def ask_question(question):
    retrieved_docs = retrieve(question)

    if not retrieved_docs:
        return {
            "question": question,
            "answer": "No relevant context found.",
            "sources": []
        }

    context = "\n\n".join([doc["text"] for doc in retrieved_docs])
    prompt = build_prompt(context, question)
    answer = generate_answer(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": retrieved_docs
    }

# ============================================================
# EVALUATION QUESTIONS (5–10 samples)
# ============================================================

evaluation_questions = [
    "What are common issues with credit cards?",
    "Why do customers complain about money transfers?",
    "What problems occur with savings accounts?",
    "What complaints are seen in personal loans?",
    "Are there delays in transaction processing?",
]

# ============================================================
# RUN EVALUATION
# ============================================================

if __name__ == "__main__":

    results = []

    print("\nRunning evaluation...\n")

    for q in evaluation_questions:
        output = ask_question(q)

        print("\n" + "=" * 70)
        print("QUESTION:", q)
        print("\nANSWER:\n", output["answer"])

        print("\nTOP SOURCES:")
        for i, src in enumerate(output["sources"][:2]):
            print(f"\nSource {i+1}:")
            print(src["text"][:300])

        results.append(output)

    print("\nTASK 3 COMPLETED SUCCESSFULLY")