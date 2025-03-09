import torch
import os
import shutil
import sys
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from data_processing import train_dataloader, val_dataloader
from transformers import AutoModel
from huggingface_hub import hf_hub_download

device = "cuda"
epochs = 15
save_path = "data/weights/"

# Функция для сохранения весов модели локально
def save_model_weights(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")

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

# Load the Hugging Face model
HF_TOKEN = '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'
path = os.path.expanduser('~/.cvlface_cache/minchul/cvlface_adaface_vit_base_webface4m')
repo_id = 'minchul/cvlface_adaface_vit_base_webface4m'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Убедитесь, что модель загружена и веса сохранены локально
model = load_model_by_repo_id(repo_id, path, HF_TOKEN)

def train(model, optimizer, criterion, epochs, train_dataloader, test_dataloader):
  os.makedirs(save_path, exist_ok=True)
  summary_loss_train = []
  summary_loss_test = []
  for epoch in tqdm(range(epochs), desc="Epochs", ncols=100):
    train_loss = 0
    train_norm_variable = 0
    model.train()
    for image0, image1, labels in train_dataloader:
      image0 = image0.to(device)
      image1 = image1.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      embed0 = model(image0)
      embed1 = model(image1)

      loss = criterion(embed0, embed1, labels)
      summary_loss_train.append(loss.item())
      train_loss += loss.item() * image0.size(0)
      train_norm_variable += image0.size(0)

      loss.backward()
      optimizer.step()

    print(f'Train loss: {train_loss/train_norm_variable}')

    model_path = os.path.join(save_path, f"model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), model_path) #СОХРАНЯЕМ ВЕСА, кошмар и треш, при обучении не сохранились, теперь сохраняю :)
    print(f"Model weights saved to {model_path}")

    test_loss = 0
    test_norm_variable = 0
    model.eval()
    with torch.no_grad():
      for image0, image1, labels in test_dataloader:
        image0 = image0.to(device)
        image1 = image1.to(device)
        labels = labels.to(device)

        embed0 = model(image0)
        embed1 = model(image1)

        loss = criterion(embed0, embed1, labels)
        summary_loss_test.append(loss.item())

        test_loss += loss.item() * image0.size(0)
        test_norm_variable += image0.size(0)

    print(f'Test loss: {test_loss/test_norm_variable}')

if __name__ == "__main__":

  for param in model.parameters():
    param.requires_grad = False

  for layer in model.model.net.feature:
      if isinstance(layer, torch.nn.BatchNorm1d):
          for param in layer.parameters():
              param.requires_grad = False  # Замораживаем параметры

    # Размораживаем Linear
  for layer in model.model.net.feature:
      for param in layer.parameters():
          param.requires_grad = True  



  for name, param in model.named_parameters():
      print(f"Параметр {name} обучаемый: {param.requires_grad}")

      optimizer = optim.Adam(model.head.parameters(), lr=3e-3)
      criterion = nn.CosineEmbeddingLoss()

      model = model.to('cuda')

      train_on_gpu = torch.cuda.is_available()

      train(model, optimizer, criterion, 20, train_dataloader, val_dataloader)
