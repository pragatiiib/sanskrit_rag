import streamlit as st
import os
from ctransformers import AutoModelForCausalLM
from rank_bm25 import BM25Okapi
import string

# --- Helper Functions ---
def simple_text_splitter(text, chunk_size=500, overlap=50):
    """
    Splits text into chunks of specified size with overlap.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        
    return chunks

def tokenize(text):
    """
    Simple tokenizer: lowercase and split by whitespace/punctuation.
    """
    text = text.lower()
    # Remove punctuation
    for char in string.punctuation:
        text = text.replace(char, ' ')
    return text.split()

@st.cache_resource
def load_resources():
    # 1. Load Data
    data_path = "data/sanskrit_stories.txt"
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}")
        return None, None, None

    with st.spinner("Indexing Sanskrit documents..."):
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            chunks = simple_text_splitter(text)
            tokenized_corpus = [tokenize(doc) for doc in chunks]
            
            # 2. Build BM25 Index
            bm25 = BM25Okapi(tokenized_corpus)
        except Exception as e:
            st.error(f"Error indexing data: {e}")
            return None, None, None
    
    # 3. Load LLM
    model_name = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    model_dir = "models"
    model_path = os.path.join(model_dir, model_name)
    
    # Create models directory if not exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Download if not present
    if not os.path.exists(model_path):
        from huggingface_hub import hf_hub_download
        with st.spinner(f"Downloading model {model_name}... (This happens only once)"):
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=model_name,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                return None, None, None

    print(f"Loading LLM from {model_path}...")
    try:
        # Initialize CTransformers directly
        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="llama",
            context_length=2048,
            max_new_tokens=512,
            temperature=0.1
        )
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None
    
    return bm25, chunks, llm

def main():
    st.set_page_config(page_title="Sanskrit RAG", layout="wide")
    st.title("🕉️ Sanskrit App (CPU)")
    st.markdown("Ask questions about the Sanskrit stories.")
    
    bm25, chunks, llm = load_resources()
    
    if bm25 is None or llm is None:
        return

    query = st.text_input("Enter your query:", placeholder="e.g., What did the foolish servant do?")
    
    if query:
        with st.spinner("Thinking..."):
            try:
                # 1. Retrieve
                tokenized_query = tokenize(query)
                top_chunks = bm25.get_top_n(tokenized_query, chunks, n=3)
                
                context = "\n\n".join(top_chunks)
                
                # 2. Construct Prompt
                prompt = f"""You are a helpful assistant. Use the following Sanskrit text context to answer the question in English.
If you don't know the answer, say you don't know.

Context:
{context}

Question: {query}
Answer:"""

                # 3. Generate
                # CTransformers 'generate' or call the object
                response = llm(prompt)
                
                st.markdown("### Answer")
                st.write(response)
                
                with st.expander("Source Documents (Context)"):
                    for i, chunk in enumerate(top_chunks):
                        st.markdown(f"**Chunk {i+1}**")
                        st.text(chunk)
                        st.divider()
                        
            except Exception as e:
                st.error(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()
