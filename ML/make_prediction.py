import os
from typing import List
import torch
import pandas as pd
from torch.nn import functional as F
from torchvision import transforms
from oml import datasets as d
from oml.inference import inference

# Load the Hugging Face model from the first code snippet
from transformers import AutoModel
from huggingface_hub import hf_hub_download
import shutil
import sys

# Путь для сохранения локальных весов
LOCAL_WEIGHTS_PATH = "data/weights/model_weights_epoch_3.pth"

# Helper function to download Hugging Face repo
def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)

# Helper function to load model from local path
def load_model_from_local_path(path, HF_TOKEN=None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model

# Helper function to download Hugging Face repo and use model
def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)

# Функция для загрузки весов модели локально
def load_model_weights(model, path, device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Model weights loaded from {path}")
    else:
        print(f"No local weights found at {path}")
    return model

# Load the Hugging Face model
HF_TOKEN = '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'
path = os.path.expanduser('~/.cvlface_cache/minchul/cvlface_adaface_vit_base_webface4m')
repo_id = 'minchul/cvlface_adaface_vit_base_webface4m'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Убедитесь, что модель загружена и веса сохранены локально

model = load_model_by_repo_id(repo_id, path, HF_TOKEN)
model = load_model_weights(model, LOCAL_WEIGHTS_PATH, device)

# Ensure the model is on the correct device
model = model.to(device).eval()

# Define the transformation for the input images
transform_test = transforms.Compose([
    transforms.Resize((112, 112)),  # Resize to the expected input size
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize as required by the model
])

# Function to create a sample submission
def create_sample_sub(pair_ids: List[str], sim_scores: List[float]):
    sub_sim_column = "similarity"
    id_column = "pair_id"
    return pd.DataFrame({id_column: pair_ids, sub_sim_column: sim_scores})

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    
    test_path = "test.csv"
    
    # Load test data
    df_test = pd.read_csv(test_path)
    test = d.ImageQueryGalleryLabeledDataset(df_test, transform=transform_test)
    
    # Perform inference using the Hugging Face model
    embeddings = inference(model, test, batch_size=32, num_workers=0, verbose=True)
    
    # Calculate cosine similarity between pairs
    e1 = embeddings[::2]
    e2 = embeddings[1::2]
    sim_scores = F.cosine_similarity(e1, e2).detach().cpu().numpy()
    
    # Prepare pair IDs for submission
    pair_ids = df_test["label"].apply(lambda x: f"{x:08d}").to_list()
    pair_ids = pair_ids[::2]
    
    # Create and save the submission file
    sub_df = create_sample_sub(pair_ids, sim_scores)
    sub_df.to_csv("data/submission.csv", index=False)
