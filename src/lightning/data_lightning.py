from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
import torch
import pytorch_lightning as pl
from torchvision import datasets, transforms
import multiprocessing


class LightningCIFAR10(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir='data'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.num_workers = multiprocessing.cpu_count()

    def setup(self, stage=None):
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        cifar_dataset = datasets.CIFAR10('data',
                                         download=True,
                                         train=True,
                                         transform=train_transform, )
        self.test_dataset = datasets.CIFAR10(
            root='data', train=False,
            download=True, transform=test_transform,
        )
        self.train_set, self.val_set = torch.utils.data.random_split(cifar_dataset, [45000, 5000])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=self.num_workers)


def load_mnist_data(batch=60):
    """
    Load Mnist data
    :param batch:
    :return: loaders for train, validation and test
    """
    test_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomCrop(28, padding=4),
        # transforms.ColorJitter(brightness=.5, hue=.3),
        # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    mnsit_dataset = datasets.MNIST('data',
                                   download=True,
                                   train=True,
                                   transform=train_transform, )
    test_dataset = datasets.MNIST(
        root='data', train=False,
        download=True, transform=test_transform,
    )
    train_set, val_set = torch.utils.data.random_split(mnsit_dataset, [55000, 5000])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch,
                                               shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch,
                                             shuffle=False, num_workers=2)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch,
                                              shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


def load_cifar10_data(batch=60):
    """
    Load Mnist data
    :param batch:
    :return: loaders for train, validation and test
    """
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_dataset = datasets.CIFAR10('data',
                                     download=True,
                                     train=True,
                                     transform=train_transform, )
    test_dataset = datasets.CIFAR10(
        root='data', train=False,
        download=True, transform=test_transform,
    )
    train_set, val_set = torch.utils.data.random_split(cifar_dataset, [45000, 5000])
    return train_set, val_set, test_dataset


if __name__ == '__main__':
    train, val, test = load_mnist_data()
