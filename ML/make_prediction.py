import os
import torch
import timm
from torch.nn import functional as F
from torchvision import transforms
from huggingface_hub import login
from typing import List
import pandas as pd
from oml import datasets as d
from oml.inference import inference

def create_sample_sub(pair_ids: List[str], sim_scores: List[float]):
    sub_sim_column = "similarity"
    id_column = "pair_id"
    return pd.DataFrame({id_column: pair_ids, sub_sim_column: sim_scores})


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")

    device = "cpu"
    test_path = "test.csv"

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

    df_test = pd.read_csv(test_path)
    test = d.ImageQueryGalleryLabeledDataset(df_test, transform=transform)
    embeddings = inference(model, test, batch_size=32, num_workers=0, verbose=True)

    e1 = embeddings[::2]
    e2 = embeddings[1::2]
    sim_scores = F.cosine_similarity(e1, e2).detach().cpu().numpy()

    pair_ids = df_test["label"].apply(lambda x: f"{x:08d}").to_list()
    pair_ids = pair_ids[::2]

    sub_df = create_sample_sub(pair_ids, sim_scores)
    sub_df.to_csv("data/submission.csv", index=False)


