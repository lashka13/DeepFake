import os
import torch
import timm
from torch.nn import functional as F
from torchvision import transforms
from huggingface_hub import login


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")

    device = "cpu"

    login(token="Введите ваш токен")
    model = timm.create_model("hf_hub:gaunernst/vit_tiny_patch8_112.arcface_ms1mv3", pretrained=True)
    state_dict = torch.load("data/weights/model_epoch_20.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    transform = transforms.Compose([
      transforms.Resize((112, 112)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
      ])
    
    transform_image1 = transform(image1)
    transform_image1 = transform(image2)

    e1 = model(image1)
    e2 = model(image2)
    sim_scores = F.cosine_similarity(e1, e2).detach().cpu().numpy()

