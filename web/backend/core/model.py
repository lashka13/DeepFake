import torch
import os
from torch.nn import functional as F
from torchvision import transforms
from transformers import AutoModel
from huggingface_hub import hf_hub_download
import shutil
import sys
from dotenv import load_dotenv


LOCAL_WEIGHTS_PATH = "core/weights/model_final_weights_20.pth"


def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, "files.txt")
    if not os.path.exists(files_path):
        hf_hub_download(
            repo_id,
            "files.txt",
            token=HF_TOKEN,
            local_dir=path,
            local_dir_use_symlinks=False,
        )
    with open(os.path.join(path, "files.txt"), "r") as f:
        files = f.read().split("\n")
    for file in [f for f in files if f] + [
        "config.json",
        "wrapper.py",
        "model.safetensors",
    ]:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(
                repo_id,
                file,
                token=HF_TOKEN,
                local_dir=path,
                local_dir_use_symlinks=False,
            )


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


# Load environment variables from .env file
load_dotenv()

# Get the token from environment variables
HF_TOKEN = os.getenv("TOKEN")

# Load the Hugging Face model
path = os.path.expanduser("~/.cvlface_cache/minchul/cvlface_adaface_vit_base_webface4m")
repo_id = "minchul/cvlface_adaface_vit_base_webface4m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Убедитесь, что модель загружена и веса сохранены локально

model = load_model_by_repo_id(repo_id, path, HF_TOKEN)
model = load_model_weights(model, LOCAL_WEIGHTS_PATH, device)

# Ensure the model is on the correct device
model = model.to(device).eval()


async def compare_images(first_image, second_image):
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    transform_image1 = transform(first_image).unsqueeze(0).to(device)  # Move to device
    transform_image2 = transform(second_image).unsqueeze(0).to(device)  # Move to device

    e1 = model(transform_image1)
    e2 = model(transform_image2)
    sim_scores = F.cosine_similarity(e1, e2).detach().cpu().numpy()

    return sim_scores
