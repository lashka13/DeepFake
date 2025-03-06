from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms


class FaceRecognitionDataset(Dataset):
  def __init__(self, pairs, mode):
    self.pairs = pairs
    self.mode = mode
    self.transform_train = transforms.Compose([
      transforms.Resize((112, 112)),
      transforms.RandomHorizontalFlip(p=0.2),  # Горизонтальный флип
      #transforms.ColorJitter(brightness=0.5, contrast=0.5),  # Изменение яркости и контраста
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
      ])
    self.transform_test = transforms.Compose([
      transforms.Resize((112, 112)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
      ])
  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, index):
    path_to_image0 = Path(f"data/train/images/{self.pairs[index][0][0]}") #надо бы поменять конечно, но работает
    path_to_image1 = Path(f"data/train/images/{self.pairs[index][0][1]}") #надо бы поменять конечно, но работает

    label = self.pairs[index][1]

    if self.mode == 'train':
      image2 = self.transform_train(Image.open(path_to_image0))
      image2 = image2.requires_grad_(True)
      image3 = self.transform_train(Image.open(path_to_image1))
      image3 = image3.requires_grad_(True)
    if self.mode == 'test':
      image2 = self.transform_test(Image.open(path_to_image0))
      image3 = self.transform_test(Image.open(path_to_image1))

    return image2, image3, label