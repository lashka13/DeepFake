import torch
import timm
from huggingface_hub import login
from dotenv import load_dotenv
import os
from torch.nn import functional as F
from torchvision import transforms

# Load environment variables from .env file
load_dotenv()

# Get the token from environment variables
TOKEN = os.getenv("TOKEN")

# Connect the model
device = "cpu"
login(token=TOKEN)
model = timm.create_model("hf_hub:gaunernst/vit_tiny_patch8_112.arcface_ms1mv3", pretrained=True)
state_dict = torch.load("core/weights/model_epoch_20.pth", map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(device).eval()


async def compare_images(first_image, second_image):
    transform = transforms.Compose([
      transforms.Resize((112, 112)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
      ])
    
    transform_image1 = transform(first_image).unsqueeze(0).to(device)  # Move to device
    transform_image2 = transform(second_image).unsqueeze(0).to(device)  # Move to device

    e1 = model(transform_image1)
    e2 = model(transform_image2)
    sim_scores = F.cosine_similarity(e1, e2).detach().cpu().numpy()
    
    print(sim_scores)
    return sim_scores


