import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/filtered_complaints.csv"
OUTPUT_DIR = "vector_store"

SAMPLE_SIZE = 12000
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# CUSTOM CHUNKING FUNCTION
# ============================================================

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Original dataset size:", len(df))

# ============================================================
# STRATIFIED SAMPLING
# ============================================================

sample_size = min(SAMPLE_SIZE, len(df))

df_sample, _ = train_test_split(
    df,
    train_size=sample_size,
    stratify=df["Product"],
    random_state=42
)

df_sample = df_sample.reset_index(drop=True)

print("Sample size:", len(df_sample))

# ============================================================
# CREATE CHUNKS
# ============================================================

all_chunks = []
metadata = []

for idx, row in df_sample.iterrows():

    text = str(row["cleaned_narrative"])
    product = row["Product"]

    chunks = chunk_text(
        text,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    for chunk in chunks:
        all_chunks.append(chunk)

        metadata.append({
            "complaint_id": idx,
            "product": product
        })

print("Total chunks created:", len(all_chunks))

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

#model = SentenceTransformer(MODEL_NAME)
model = SentenceTransformer(MODEL_NAME, device="cpu")

# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("Generating embeddings...")

embeddings = model.encode(
    all_chunks,
    show_progress_bar=True,
    convert_to_numpy=True
)

embedding_dim = embeddings.shape[1]

# ============================================================
# BUILD FAISS INDEX
# ============================================================

print("Building FAISS index...")

index = faiss.IndexFlatL2(embedding_dim)

index.add(
    np.array(embeddings).astype("float32")
)

# ============================================================
# SAVE VECTOR STORE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

faiss.write_index(
    index,
    os.path.join(OUTPUT_DIR, "faiss_index.bin")
)

with open(
    os.path.join(OUTPUT_DIR, "metadata.pkl"),
    "wb"
) as f:
    pickle.dump(metadata, f)

with open(
    os.path.join(OUTPUT_DIR, "chunks.pkl"),
    "wb"
) as f:
    pickle.dump(all_chunks, f)

print("Vector store saved successfully.")

# ============================================================
# SUMMARY
# ============================================================

print(f"Total sampled complaints: {len(df_sample)}")
print(f"Total chunks: {len(all_chunks)}")
print(f"Embedding dimension: {embedding_dim}")