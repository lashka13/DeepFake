import os
import random
import json
from itertools import combinations
from sklearn.model_selection import train_test_split
from FaceRecognitionDataset import FaceRecognitionDataset
from torch.utils.data import DataLoader

people = {people: len(os.listdir(f'data/train/images/{people}')) for people in os.listdir('data/train/images')}

meta_path = "data/train/meta.json"

with open(meta_path, "r") as f:
    meta_data = json.load(f)


real_images = {}  #словарь для реальных изображений
fake_images = {}  #словарь для синтетических изображений

for img_path, label in meta_data.items():
    person_id, img_name = img_path.split("/")  #разделяем индекс человека и номер изображения

    if label == 0:
        if person_id not in real_images:
            real_images[person_id] = []
        real_images[person_id].append(img_path)
    else:
        if person_id not in fake_images:
            fake_images[f'{person_id}_fake'] = []
        fake_images[f'{person_id}_fake'].append(img_path)

positive_pairs = []
negative_pairs = []

#создаем пары
for person, images in real_images.items():
    if len(images) > 1:
        positive_pairs += [(i,1) for i in combinations(images, 2)] # Все возможные пары для одного человека
# пары без повторений и учета порядка - уникальные
# Создаем негативные пары
people = list(real_images.keys())
people_fake = list(fake_images.keys())
for _ in range(len(positive_pairs)):
    if random.choice([True, False]):
        p1, p2 = random.sample(people, 2)
        img1 = random.choice(real_images[p1])
        img2 = random.choice(real_images[p2])
        negative_pairs.append(((img1, img2), -1))
    else:
        p1 = random.choice(people)
        img1 = random.choice(real_images[p1])
        img2 = random.choice(fake_images[f'{p1}_fake'])
        negative_pairs.append(((img1, img2), -1))

print(f"✔ Создано {len(positive_pairs)} позитивных пар и {len(negative_pairs)} негативных пар.")
all_pairs = positive_pairs[:15000] + negative_pairs[:22000]
test_pairs = positive_pairs[15000:15600] + negative_pairs[22000:22800]
all_paits = random.shuffle(all_pairs)

train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.25, random_state=42) #разобьем выборку на треин и валидацию


train_dataset = FaceRecognitionDataset(train_pairs, 'train')
val_dataset = FaceRecognitionDataset(val_pairs, 'test')
test_dataset = FaceRecognitionDataset(test_pairs, 'test') # для оценки метрики

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0)