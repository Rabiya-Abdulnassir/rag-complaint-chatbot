import faiss
import numpy as np
import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ============================================================
# CONFIG
# ============================================================

VECTOR_STORE_PATH = "vector_store"
PARQUET_PATH = "data/complaint_embeddings.parquet"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

print("Starting RAG pipeline...")

# ============================================================
# LOAD PARQUET (THIS IS YOUR REAL DATA SOURCE)
# ============================================================

if not os.path.exists(PARQUET_PATH):
    raise FileNotFoundError("Parquet file not found. Put it in /data folder.")

df = pd.read_parquet(PARQUET_PATH)

# EXPECTED COLUMNS (adjust if needed):
# text OR complaint_text OR chunk
TEXT_COL = "text" if "text" in df.columns else df.columns[0]

texts = df[TEXT_COL].astype(str).tolist()

print(f"Loaded {len(texts)} rows from parquet")

# ============================================================
# LOAD FAISS INDEX
# ============================================================

faiss_index_path = os.path.join(VECTOR_STORE_PATH, "faiss_index.bin")

if not os.path.exists(faiss_index_path):
    raise FileNotFoundError("FAISS index not found")

index = faiss.read_index(faiss_index_path)

print("FAISS index loaded")

# ============================================================
# EMBEDDING MODEL
# ============================================================

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
    cache_folder="./hf_cache"
)

print("Embedding model loaded")

# ============================================================
# RETRIEVER (FIXED TO USE PARQUET TEXTS)
# ============================================================

def retrieve(query, top_k=TOP_K):

    query_vec = model.encode(query, convert_to_numpy=True)
    query_vec = np.array([query_vec]).astype("float32")

    distances, indices = index.search(query_vec, top_k)

    results = []

    for i in indices[0]:
        if i == -1:
            continue
        if i < len(texts):
            results.append({
                "text": texts[i],
                "metadata": {"row_id": int(i)}
            })

    return results

# ============================================================
# PROMPT
# ============================================================

def build_prompt(context, question):
    return f"""
You are a financial complaint analyst.

Answer only using the context.

If not found, say: I don't have enough information.

Context:
{context}

Question:
{question}

Answer:
""".strip()

# ============================================================
# LLM (FIXED SEQ2SEQ USAGE)
# ============================================================

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_answer(prompt):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=120
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ============================================================
# RAG PIPELINE
# ============================================================

def ask_question(question):

    docs = retrieve(question)

    if not docs:
        return {
            "question": question,
            "answer": "No relevant context found",
            "sources": []
        }

    context = "\n\n".join([d["text"][:400] for d in docs])

    prompt = build_prompt(context, question)

    answer = generate_answer(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": docs
    }

# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    questions = [
        "What are common issues with credit cards?",
        "Why do customers complain about money transfers?",
        "What problems occur with savings accounts?",
        "What complaints are seen in personal loans?",
        "Are there delays in transaction processing?"
    ]

    for q in questions:
        out = ask_question(q)

        print("\n" + "="*70)
        print("Q:", q)
        print("A:", out["answer"])

        print("\nSources:")
        for s in out["sources"][:2]:
            print("-", s["text"][:200])

    print("\nTASK 3 COMPLETED SUCCESSFULLY")