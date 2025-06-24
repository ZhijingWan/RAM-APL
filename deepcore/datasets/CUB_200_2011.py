import PIL
from PIL import Image
import os
import torch
import pandas as pd
from torchvision import transforms

def CUB_200_2011(data_path: str):
    """
    Load clean version of CUB-200-2011 dataset.

    Args:
        data_path: Root directory containing the dataset.

    Returns:
        Tuple: channel, image size, number of classes, class names, mean, std,
               train dataset, test dataset.
    """
    channel = 3
    im_size = (448, 448)
    num_classes = 200
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
                    transforms.Resize(256), #512
                    transforms.RandomCrop(224), #448
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                    ])

    test_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
                        ])
    root = os.path.join(data_path, 'CUB_200_2011/CUB_200_2011')
    dst_train = ImageLoader(root, transform = train_transform, train = True)
    dst_test = ImageLoader(root, transform = test_transform, train = False)

    class_names = dst_train.classes
    dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, train=False, loader=pil_loader, tta=None):
        img_folder = os.path.join(root, "images")
        img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,  names=['idx', 'label'])
        train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,  names=['idx', 'train_flag'])
        data = pd.concat([img_paths, img_labels, train_test_split], axis=1)
        data = data[data['train_flag'] == train]
        data['label'] = data['label'] - 1

        imgs = data.reset_index(drop=True)
        self.targets = data['label'].values

        classes_name = pd.read_csv(os.path.join(root, "classes.txt"), sep=" ", header=None,  names=['idx', 'classes'])
        self.classes = classes_name['classes'].values

        if len(imgs) == 0:
            raise(RuntimeError("no csv file"))

        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.tta = tta

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.imgs.iloc[index]
        file_path = item['path']
        target = item['label']

        img = self.loader(os.path.join(self.root, file_path))

        if self.tta is None:
            img = self.transform(img)
        elif self.tta == 'flip':
            img_1 = self.transform(img)
            img_2 = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            img_2 = self.transform(img_2)
            img = torch.stack((img_1, img_2), dim=0)
        else:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)