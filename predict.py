import torch
import timm
import pandas as pd
from torchvision import transforms

from oml import datasets as d
from oml.inference import inference
from oml.metrics import calc_retrieval_metrics_rr

from huggingface_hub import login
from oml.retrieval import RetrievalResults, AdaptiveThresholding

device = "cpu"

if __name__ == "__main__":
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

    df_test = pd.read_csv("test.csv")
    test = d.ImageQueryGalleryLabeledDataset(df_test, transform=transform)

    def predict():
        embeddings = inference(model, test, batch_size=32, num_workers=6, verbose=True)
        rr = RetrievalResults.from_embeddings(embeddings, test, n_items=10)
        rr = AdaptiveThresholding(n_std=2).process(rr)
        rr.visualize(query_ids=[2, 1], dataset=test, show=True)
        results = calc_retrieval_metrics_rr(rr, map_top_k=(10,), cmc_top_k=(1, 5, 10))

        for metric_name in results.keys():
            for k, v in results[metric_name].items():
                print(f"{metric_name}@{k}: {v.item()}")

    predict()
