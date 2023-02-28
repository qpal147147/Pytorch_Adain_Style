from pathlib import Path

import cv2
import torch
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)

    inv_trans = T.Normalize((-mean/std).tolist(), (1.0/std).tolist())
    return inv_trans(x)

def train_transforms(img_size, crop_size):
    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(img_size),
        T.RandomCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transforms

def test_transforms(img_size, crop_size):
    transforms = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    return transforms

def create_dataloader(contents_path, styles_path, img_size=512, crop_size=256, batch_size=8, shuffle=True, num_workers=8, training=True):
    assert img_size >= crop_size, "image size < crop size !"

    if training:
        dataset = LoadImage(contents_path, styles_path, img_size, crop_size, train_transforms)
    else:
        dataset = LoadImage(contents_path, styles_path, img_size, crop_size, test_transforms)
        
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)

class LoadImage(Dataset):
    def __init__(self, contents_path, styles_path, img_size, crop_size, transforms):
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.crop_size = crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)
        self.transform = transforms(self.img_size, self.crop_size)

        c_f = Path(contents_path)
        s_f = Path(styles_path)

        if c_f.is_dir() and s_f.is_dir():
            self.contents = [x for x in c_f.glob("*.*") if x.is_file()]
            self.styles = [x for x in s_f.glob("*.*") if x.is_file()]
        elif c_f.is_file() and s_f.is_file():
            self.contents = [c_f]
            self.styles = [s_f]

        assert len(self.contents) == len(self.styles), "content images != style images "

    def __getitem__(self, index):
        c_path, s_path = self.contents[index], self.styles[index]

        content = cv2.cvtColor(cv2.imread(str(c_path)), cv2.COLOR_BGR2RGB)
        style = cv2.cvtColor(cv2.imread(str(s_path)), cv2.COLOR_BGR2RGB)

        content = self.transform(content)
        style = self.transform(style)

        return content, style

    def __len__(self):
        return len(self.contents)

if __name__ == "__main__":
    dataset = LoadImage("./content", "./style", 512, 256)
    print(len(dataset))

