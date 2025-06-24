from torchvision import datasets, transforms
import torch
import os

# Oxford-IIIT Pet
def Pet(data_path: str):
	"""
    Load clean version of Oxford-IIIT Pet dataset. <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        data_path: Root path of the dataset.

    Returns:
        Tuple: channel, image size, class count, class names, mean, std,
               train dataset, test dataset.
    """
	channel = 3
	im_size = (224, 224)
	num_classes = 37
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	DATASET = 'pet'
	data_path = os.path.join(data_path, DATASET)
	transform = transforms.Compose([transforms.Resize(256),
									transforms.RandomResizedCrop(224, antialias=True), 
									transforms.ToTensor(), 
									transforms.Normalize(mean=mean, std=std)])
	transform_test = transforms.Compose([transforms.Resize(256),
										transforms.CenterCrop(224),
										transforms.ToTensor(),
										transforms.Normalize(mean=mean, std=std)])


	dst_train = datasets.ImageFolder(os.path.join(data_path, 'trainval'), transform)
	dst_test = datasets.ImageFolder(os.path.join(data_path, 'test'), transform_test)
	class_names = dst_train.classes
	dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
	dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)

	return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test