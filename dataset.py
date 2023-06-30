import numpy as np
import torch
import torchvision 
from torch.utils.data import DataLoader

torch.manual_seed(1)


class Cifar10Dataset(torchvision.datasets.CIFAR10):
    """
    Custom Dataset Class

    """

    def __init__(self, root="../data", train=True, download=True, transforms=None):
        """Initialize Dataset

        Args:
            dataset (Dataset): Pytorch Dataset instance
            transforms (Transform.Compose, optional): Tranform function instance. Defaults to None.
        """
        super().__init__(root=root, train=train, download=download, transform=transforms)
        self.transforms = transforms


    def __getitem__(self, idx):
        """Get an item form dataset

        Args:
            idx (int): id of item in dataset

        Returns:
            (tensor, int): Return tensor of transformer image, label
        """
        # Read Image and Label
        image, label = self.data[idx], self.targets[idx]

        # image = np.array(image)

        # Apply Transforms
        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        return (image, label)


def get_loader(train_data, test_data, batch_size=64, use_cuda=True):
    """Get instance of tran and test loaders

    Args:
        train_transform (Transform): Instance of transform function for training
        test_transform (Transform): Instance of transform function for validation
        batch_size (int, optional): batch size to be uised in training. Defaults to 64.
        use_cuda (bool, optional): Enable/Disable Cuda Gpu. Defaults to True.

    Returns:
        (DataLoader, DataLoader): Get instance of train and test data loaders
    """
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(train_data,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(test_data,
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader