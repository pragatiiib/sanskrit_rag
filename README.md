# Sanskrit RAG System (CPU-based)

This project implements a Retrieval-Augmented Generation (RAG) system specialized for Sanskrit documents, designed to run entirely on CPU.
Due to environment constraints with Torch on Windows, this version uses **BM25** for retrieval (sparse vector search) which is fast and robust for CPU usage.

## Architecture

1.  **Ingestion & Retrieval**:
    *   Loads Sanskrit text documents (UTF-8) from `data/`.
    *   Splits text into chunks.
    *   Uses **BM25** (Best Matching 25) algorithm for retrieval. This runs entirely in-memory and requires no heavy GPU libraries.

2.  **Generation**:
    *   A quantized Large Language Model (LLM) running locally on CPU generates answers based on the retrieved context.
    *   Library: `ctransformers` (GGML/GGUF support).
    *   Model: `TinyLlama-1.1B-Chat-v1.0-GGUF`.

## Prerequisites

- Python 3.10+

## Setup

1.  **Install dependencies**:
    ```bash
    py -m pip install -r requirements.txt
    ```

2.  **Download the LLM model**:
    ```bash
    py src/download_model.py
    ```

3.  **Run the Query Interface**:
    ```bash
    py -m streamlit run src/app.py
    ```

## Usage

- The app will open in your browser (usually `http://localhost:8501`).
- Enter a query about the Sanskrit stories (e.g., "What happened to the foolish servant?").
