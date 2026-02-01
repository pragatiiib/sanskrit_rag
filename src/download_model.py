import os
from huggingface_hub import hf_hub_download

def download_model():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    print(f"Downloading {filename} from {repo_id}...")
    model_path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        local_dir=model_dir, 
        local_dir_use_symlinks=False
    )
    print(f"Model downloaded to {model_path}")

if __name__ == "__main__":
    download_model()
