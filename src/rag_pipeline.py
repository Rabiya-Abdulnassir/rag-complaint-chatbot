import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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

if not os.path.exists(faiss_index_path):
    raise FileNotFoundError("FAISS index not found")

if not os.path.exists(chunks_path):
    raise FileNotFoundError("Chunks file not found")

if not os.path.exists(metadata_path):
    raise FileNotFoundError("Metadata file not found")

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

try:
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder="./hf_cache"
    )

    print("Embedding model loaded successfully")

except Exception as e:
    print("ERROR loading embedding model:", e)
    raise

# ============================================================
# RETRIEVER FUNCTION
# ============================================================

def retrieve(query, top_k=TOP_K):

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    query_embedding = np.array(
        [query_embedding]
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

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
You are a financial analyst assistant.

Use ONLY the information provided in the context.

Provide a concise answer.

If the answer cannot be found in the context, say:
I don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

# ============================================================
# LOAD FLAN-T5
# ============================================================

print("Loading LLM...")

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-base"
)

llm_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base"
)

print("LLM loaded successfully")

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_answer(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer

# ============================================================
# RAG PIPELINE
# ============================================================

def ask_question(question):

    retrieved_docs = retrieve(question)

    if len(retrieved_docs) == 0:
        return {
            "question": question,
            "answer": "No relevant context found.",
            "sources": []
        }

    context = "\n\n".join(
        [doc["text"][:500] for doc in retrieved_docs]
    )

    prompt = build_prompt(
        context,
        question
    )

    answer = generate_answer(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": retrieved_docs
    }

# ============================================================
# EVALUATION SET
# ============================================================

evaluation_questions = [
    "What are common issues with credit cards?",
    "Why do customers complain about money transfers?",
    "What problems occur with savings accounts?",
    "What complaints are seen in personal loans?",
    "Are there delays in transaction processing?"
]

# ============================================================
# RUN EVALUATION
# ============================================================

if __name__ == "__main__":

    print("\nRunning evaluation...\n")

    results = []

    for q in evaluation_questions:

        output = ask_question(q)

        print("\n" + "=" * 70)
        print("QUESTION:", q)

        print("\nANSWER:")
        print(output["answer"])

        print("\nTOP SOURCES:")

        for i, src in enumerate(output["sources"][:2]):

            print(f"\nSource {i + 1}:")
            print(src["text"][:300])

        results.append(output)

    print("\nTASK 3 COMPLETED SUCCESSFULLY")