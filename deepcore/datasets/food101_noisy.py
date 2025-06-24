import os
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision import transforms
from typing import Optional, Callable, Tuple, Any

from .utils import noisify

def FOOD101_NOISY(data_path: str, noise_type: str, noise_rate: float):
    """
    Load noisy version of FOOD101 dataset.
    
    Args:
        data_path: Root path of the dataset.
        noise_type: Type of label noise ('symmetric', 'asymmetric', etc.).
        noise_rate: Rate of label corruption (0 to 1).

    Returns:
		Tuple including dataset metadata and train/test splits with noisy labels.
    """
    channel = 3
    im_size = (224, 224)
    num_classes = 101
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([transforms.RandomResizedCrop(224, antialias=True), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=mean, std=std)])
    transform_test = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])

    dst_train = Food101_Noisy(root=data_path,
							train=True, 
							transform=transform_train, 
							download=False,
							noise_type=noise_type,
							noise_rate=noise_rate
							)
    dst_test = Food101_Noisy(root=data_path, 
							train=False, 
							transform=transform_test, 
							download=False,
							noise_type=noise_type,
							noise_rate=noise_rate
							)
    class_names = dst_train.classes
    dst_train.targets = torch.tensor(dst_train.train_noisy_labels, dtype=torch.long)
    dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)

    noise_or_not_train = dst_train.noise_or_not #numpy

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, noise_or_not_train

class Food101_Noisy(VisionDataset):
    """
    Food-101 Dataset (https://www.vision.ee.ethz.ch/datasets_extra/food-101/).
    Args:
        root (str): Root directory where 'food-101' folder exists or will be downloaded.
        split (str, optional): 'train' or 'test' split. Default: 'train'.
        transform (callable, optional): A function/transform to apply to the image.
        target_transform (callable, optional): A function/transform to apply to the label.
        download (bool, optional): If True, downloads the dataset. Default: False.
    """
    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        noise_type='clean', 
        noise_rate=0.2,
        random_state=0
    ):
        # 校验 transform 是否可调用
        if transform is not None and not callable(transform):
            raise TypeError(f"transform must be callable, got {type(transform)}")

        assert 0 <= noise_rate <= 1, "Noise rate must be in [0, 1]"

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._base_folder = os.path.join(root, "food-101")
        self._meta_folder = os.path.join(self._base_folder, "meta")
        self._images_folder = os.path.join(self._base_folder, "images")

        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.train = train

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it.")

        self.targets = []
        self._image_files = []
        
        # Load class names
        with open(os.path.join(self._meta_folder, "classes.txt"), "r") as f:
            self.classes = [line.strip() for line in f]
            self.nb_classes = len(self.classes)
        print('self.nb_classes=', self.nb_classes)
        
        # Load image paths and labels
        split_file = os.path.join(self._meta_folder, "train.txt" if train else "test.txt")
        with open(split_file, "r") as f:
            for line in f:
                class_name, img_name = line.strip().split("/")
                img_path = os.path.join(self._images_folder, f"{class_name}/{img_name}.jpg")
                self._image_files.append(img_path)
                self.targets.append(self.classes.index(class_name))

        if self.train and (noise_type !='clean'):
            # noisify train data
            self.train_labels=np.asarray([[self.targets[i]] for i in range(len(self.targets))])
            self.train_noisy_labels, self.actual_noise_rate = noisify(dataset='Food-101', train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state, nb_classes=self.nb_classes)
            self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
            clean_labels = [i[0] for i in self.train_labels]
            self.noise_or_not = np.array(self.train_noisy_labels) == np.array(clean_labels)

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx):
        image = Image.open(self._image_files[idx]).convert("RGB")

        if self.train and self.noise_type !='clean':
            label = self.train_noisy_labels[idx]
        elif self.train:
            label = self.train_labels[idx]
        else:
            label = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        return os.path.isdir(self._base_folder) and os.path.exists(
            os.path.join(self._meta_folder, "classes.txt")
        )

    def _download(self):
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, self.root, md5=self._MD5)