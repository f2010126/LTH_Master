import pytorch_lightning as pl
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


class Custom_CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = 'data'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ])

        self.test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )
        self.save_hyperparameters()


    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # val and train loaders
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.train_transforms)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)
