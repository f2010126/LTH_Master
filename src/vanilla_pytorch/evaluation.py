import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(logits, labels):
    preds = torch.argmax(logits)
    return torch.sum(preds == labels) / len(labels)


def eval_fn(model, loader, device, criterion, train=False):
    """
    Evaluation method
    :param model: model to evaluate
    :param loader: data loader for either training or testing set
    :param device: torch device
    :param criterion: loss criterion
    :param train: boolean to indicate if training or test set is used
    :return: accuracy on the data
    """
    score = AverageMeter()
    losses = AverageMeter()
    model.eval()
    total_correct = 0

    t = tqdm(loader)
    with torch.no_grad():  # no gradient needed
        for images, labels in t:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            acc = accuracy(outputs, labels)
            loss = criterion(outputs, labels)
            total_correct += (outputs.argmax(dim=1) == labels).sum()
            n = images.size(0)
            score.update(acc.item(), n)
            losses.update(loss.item(), n)

            # t.set_description('(=> Test) Score: {:.4f}'.format(score.avg))

    return total_correct.item() / len(loader.dataset), losses.avg


def eval_model(model, device, saved_model_file):
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models', saved_model_file)))
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = datasets.MNIST(
        root='data', train=False,
        download=True, transform=test_transform,
    )

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=60,
                             shuffle=False)

    score = eval_fn(model, test_loader, device, criterion=torch.nn.CrossEntropyLoss().to(device), train=False)

    print('Avg accuracy:', str(score * 100) + '%')
