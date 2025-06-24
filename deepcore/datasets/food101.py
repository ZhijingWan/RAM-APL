from torchvision import datasets, transforms
import torch
import os


def FOOD101(data_path: str):
    """
    Load clean version of Food-101 dataset.
    """
    channel = 3
    im_size = (224, 224)
    num_classes = 101
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    DATASET = 'food-101/default_train_test'
    data_path = os.path.join(data_path, DATASET)
    transform = transforms.Compose([transforms.RandomResizedCrop(224, antialias=True), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=mean, std=std)])
    transform_test = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])


    dst_train = datasets.ImageFolder(os.path.join(data_path, 'train'), transform)
    dst_test = datasets.ImageFolder(os.path.join(data_path, 'test'), transform_test)
    class_names = dst_train.classes

    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test