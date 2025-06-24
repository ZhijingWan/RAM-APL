import os
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision import transforms
import torch
from typing import Optional, Callable, Union, Sequence, Tuple, Any

from .utils import download_url, check_integrity, noisify

def Pet_NOISY(data_path: str, noise_type: str, noise_rate: float):
	"""
    Load noisy version of Oxford-IIIT Pet dataset.
    
    Args:
        data_path: Root path of the dataset.
        noise_type: Type of label noise ('symmetric', 'asymmetric', etc.).
        noise_rate: Rate of label corruption.

    Returns:
		Tuple including dataset metadata and train/test splits with noisy labels.
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

	dst_train = OxfordIIITPet_Noisy(root=data_path, 
							train=True, 
							transform=transform, 
							download=False,
							noise_type=noise_type,
							noise_rate=noise_rate
							)
	dst_test = OxfordIIITPet_Noisy(root=data_path, 
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

class OxfordIIITPet_Noisy(VisionDataset):
	"""`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

	Args:
		root (string): Root directory of the dataset.
		split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
		target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
			``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

				- ``category`` (int): Label for one of the 37 pet categories.
				- ``segmentation`` (PIL image): Segmentation trimap of the image.

			If empty, ``None`` will be returned as target.

		transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
			version. E.g, ``transforms.RandomCrop``.
		target_transform (callable, optional): A function/transform that takes in the target and transforms it.
		download (bool, optional): If True, downloads the dataset from the internet and puts it into
			``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
	"""

	_RESOURCES = (
		("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
		("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
	)
	_VALID_TARGET_TYPES = ("category", "segmentation")

	def __init__(
		self,
		root: str,
		train: bool = True,
		transform: Optional[Callable] = None,
		target_types: Union[Sequence[str], str] = "category",
		target_transform: Optional[Callable] = None,
		download: bool = False,
		noise_type='clean', 
		noise_rate=0.2, 
		random_state=0,
	):
		# 校验 transform 是否可调用
		if transform is not None and not callable(transform):
			raise TypeError(f"transform must be callable, got {type(transform)}")
			
		assert 0 <= noise_rate <= 1, "Noise rate must be in [0, 1]"

		split = "trainval" if train else "test"
		self.train = train
		self.noise_type = noise_type

		if isinstance(target_types, str):
			target_types = [target_types]
		self._target_types = [target_type for target_type in target_types]

		super().__init__(root, transform=transform, target_transform=target_transform)

		self._images_folder = os.path.join(self.root, "images")
		self._anns_folder = os.path.join(self.root, "annotations")
		self._segs_folder = os.path.join(self.root, "trimaps")

		if download:
			self._download()

		if not self._check_exists():
			raise RuntimeError("Dataset not found. You can use download=True to download it")

		image_ids = []
		self.targets = []
		with open(os.path.join(self._anns_folder, f"{split}.txt")) as file:
			for line in file:
				image_id, label, *_ = line.strip().split(' ')
				image_ids.append(image_id)
				self.targets.append(int(label) - 1)

		self.nb_classes = len(set(self.targets))
		print('self.nb_classes=', self.nb_classes)

		if train and (noise_type !='clean'):
			# noisify train data
				self.train_labels=np.asarray([[self.targets[i]] for i in range(len(self.targets))])
				self.train_noisy_labels, self.actual_noise_rate = noisify(dataset='OxfordIIITPet', train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state, nb_classes=self.nb_classes)
				self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
				clean_labels=[i[0] for i in self.train_labels]
				self.noise_or_not = np.array(self.train_noisy_labels) == np.array(clean_labels)

		self.classes = [
			" ".join(part.title() for part in raw_cls.split("_"))
			for raw_cls, _ in sorted(
				{(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self.targets)},
				key=lambda image_id_and_label: image_id_and_label[1],
			)
		]
		self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

		self._images = [os.path.join(self._images_folder, f"{image_id}.jpg") for image_id in image_ids]
		self._segs = [os.path.join(self._segs_folder, f"{image_id}.png") for image_id in image_ids]

	def __len__(self) -> int:
		return len(self._images)

	def __getitem__(self, idx: int) -> Tuple[Any, Any]:
		image = Image.open(self._images[idx]).convert("RGB")
		target: Any = []

		for target_type in self._target_types:
			if target_type == "category":
				if self.train and self.noise_type !='clean':
					target.append(self.train_noisy_labels[idx])
				elif self.train:
					target.append(self.train_labels[idx])
				else:
					target.append(self.targets[idx])
			elif target_type == "segmentation":
				target.append(Image.open(self._segs[idx]))

		if not target:
			target = None
		elif len(target) == 1:
			target = target[0]
		else:
			target = tuple(target)

		if self.transform:
			image = self.transform(image)

		return image, target

	def _check_exists(self) -> bool:
		for folder in (self._images_folder, self._anns_folder):
			if not (os.path.exists(folder) and os.path.isdir(folder)):
				return False
		else:
			return True

	def _download(self) -> None:
		if self._check_exists():
			return

		for url, md5 in self._RESOURCES:
			download_and_extract_archive(url, download_root=str(self._RESOURCES), md5=md5)