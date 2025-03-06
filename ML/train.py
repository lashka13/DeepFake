import torch
import timm
import os
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from huggingface_hub import login
from data_processing import train_dataloader, val_dataloader

device = "cpu"
epochs = 15
save_path = "data/weights/"

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
    login(token="Введите ваш токен")
    model = timm.create_model("hf_hub:gaunernst/vit_tiny_patch8_112.arcface_ms1mv3", pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.head = torch.nn.Linear(192,512, bias=True)

    for name, param in model.named_parameters():
        print(f"Параметр {name} обучаемый: {param.requires_grad}")

    optimizer = optim.Adam(model.head.parameters(), lr=3e-3)
    criterion = nn.CosineEmbeddingLoss()

    model = model.to('cpu')

    train_on_gpu = torch.cuda.is_available()

    train(model, optimizer, criterion, 30, train_dataloader, val_dataloader)