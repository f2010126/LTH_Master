import torch
from torchvision import datasets, transforms


def load_data():
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=60,
                                               shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=60,
                                             shuffle=False, num_workers=2)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=60,
                                              shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train, val, test = load_data()
    print("")
