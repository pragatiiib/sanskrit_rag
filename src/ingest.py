import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest():
    # 1. Load Data
    data_path = "data/sanskrit_stories.txt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print("Loading documents...")
    loader = TextLoader(data_path, encoding='utf-8')
    documents = loader.load()

    # 2. Split Text
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Generate Embeddings
    print("Initializing embeddings model (this may take a while)...")
    # Using a multilingual model suitable for Sanskrit/English
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    # 4. Create Vector Store
    print("creating vector store...")
    db = FAISS.from_documents(chunks, embeddings)

    # 5. Save Vector Store
    save_path = "faiss_index"
    db.save_local(save_path)
    print(f"Vector store saved to {save_path}/")

if __name__ == "__main__":
    ingest()
