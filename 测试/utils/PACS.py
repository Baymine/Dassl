from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "E:/Anaconda/Library/home/lw/lw/data"

class PACS(Dataset):
    def __init__(self, root_path, domain, train=True, transform=None, target_transform=None):
        self.root = f"{root_path}/{domain}"
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        label_name_list = os.listdir(self.root)
        self.label = []
        self.data = []

        if not os.path.exists(f"{root_path}/precessed"):
            os.makedirs(f"{root_path}/precessed")
        if os.path.exists(f"{root_path}/precessed/{domain}_data.pt") and os.path.exists(
                f"{root_path}/precessed/{domain}_label.pt"):
            print(f"Load {domain} data and label from cache.")
            self.data = torch.load(f"{root_path}/precessed/{domain}_data.pt")
            self.label = torch.load(f"{root_path}/precessed/{domain}_label.pt")
        else:
            print(f"Getting {domain} datasets")
            for index, label_name in enumerate(label_name_list):
                label_name_2_index = {
                    'dog': 0,
                    'elephant': 1,
                    'giraffe': 2,
                    'guitar': 3,
                    'horse': 4,
                    'house': 5,
                    'person': 6,
                }
                images_list = os.listdir(f"{self.root}/{label_name}")
                for img_name in images_list:

                    assert os.path.isfile({self.root}/{label_name}/{img_name})

                    img = Image.open(f"{self.root}/{label_name}/{img_name}").convert('RGB')
                    img = np.array(img)
                    self.label.append(label_name_2_index[label_name])
                    if self.transform is not None:
                        img = self.transform(img)
                    self.data.append(img)
            self.data = torch.stack(self.data)
            self.label = torch.tensor(self.label, dtype=torch.long)
            torch.save(self.data, f"{root_path}/precessed/{domain}_data.pt")
            torch.save(self.label, f"{root_path}/precessed/{domain}_label.pt")

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def get_pacs_domain(root_path=f"{DATA_PATH}/PACS", domain='art_painting', verbose=False):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Resize((224, 224)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    all_data = PACS(root_path, domain, transform=transform)
    # train:test=8:2
    x_train, x_test, y_train, y_test = train_test_split(all_data.data.numpy(), all_data.label.numpy(),
                                                        test_size=0.20, random_state=42)

    return x_train, y_train, x_test, y_test
