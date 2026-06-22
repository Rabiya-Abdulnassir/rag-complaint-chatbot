# RAG Complaint Chatbot

## Overview

This project is a Retrieval-Augmented Generation (RAG) system built using the CFPB Consumer Complaint dataset. It enables semantic search and question answering over real-world financial complaint narratives using vector embeddings and large language models.

The system includes data preprocessing, exploratory data analysis, embedding generation, vector database creation, and a chatbot interface.

---

## Project Objectives

- Perform large-scale complaint data preprocessing and cleaning
- Conduct exploratory data analysis on complaint patterns
- Convert text into semantic embeddings for retrieval
- Build a FAISS or ChromaDB vector database
- Implement a Retrieval-Augmented Generation (RAG) pipeline
- Develop an interactive chatbot interface


## Task 1: Data Preprocessing and EDA

### Objective
Understand dataset structure and prepare clean text for downstream NLP tasks.

### Steps
- Loaded CFPB dataset using chunk-based processing
- Performed exploratory data analysis:
  - Complaint distribution across products
  - Narrative length analysis
  - Missing and empty narrative detection
- Filtered dataset to include:
  - Credit Card
  - Personal Loan
  - Savings Account
  - Money Transfer
- Applied text preprocessing:
  - Lowercasing
  - Removal of special characters
  - Removal of boilerplate text
  - Whitespace normalization

### Output
Cleaned dataset saved to:

---

## Task 2: Text Chunking and Embedding

### Objective
Prepare data for semantic retrieval.

### Steps
- Created stratified sample (10,000–15,000 records)
- Implemented text chunking with controlled overlap
- Generated embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Stored embeddings in FAISS or ChromaDB
- Stored metadata (complaint ID, product category)

### Output

---

## Task 3: RAG Pipeline

### Objective
Build retrieval-augmented question answering system.

### Pipeline
- Embed user query using same embedding model
- Retrieve top-k similar chunks (k=5)
- Pass retrieved context + query to LLM
- Generate grounded response based on context

### Prompt Template
You are a financial analyst assistant.

Use the following complaint excerpts to answer the question.

If the answer is not available in the context, clearly state that you do not have enough information.

Context:
{context}

Question:
{question}

Answer:

### Evaluation
- Tested on 5–10 representative queries
- Evaluated on relevance, accuracy, and grounding quality

---

## Task 4: Interactive Chat Interface

### Objective
Provide user-friendly chatbot interface.

### Features
- Text input for queries
- AI-generated responses
- Display of retrieved sources
- Clear/reset functionality
- Optional streaming responses

### Implementation
Built using Streamlit or Gradio.

---

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- SentenceTransformers
- FAISS / ChromaDB
- Hugging Face Transformers / LLM APIs
- Streamlit / Gradio

---

## Key Learnings

- Handling large datasets using chunk processing
- Importance of text cleaning for NLP pipelines
- Designing embedding-based retrieval systems
- Vector database implementation
- Prompt engineering for LLM grounding
- Building end-to-end RAG systems

---

## Future Improvements

- Hybrid retrieval (BM25 + vector search)
- Advanced reranking models
- Multi-turn conversational memory
- Improved evaluation metrics
- Cloud deployment (AWS / Hugging Face Spaces)

---

## Author

Developed as a Retrieval-Augmented Generation project using real-world financial complaint data.

