import torch
from tqdm import tqdm
import time
from evaluation import AverageMeter, accuracy
import torch.nn.functional as F


def train():
    # adam
    optimizer = torch.optim.Adam([], lr=0)


def train_fn(model, optimizer, criterion, loader, device, train=True):
    """
  Training method
  :param model: model to train
  :param optimizer: optimization algorithm
  :param criterion: loss function
  :param loader: data loader for either training or testing set
  :param device: torch device
  :param train: boolean to indicate if training or test set is used
  :return: (accuracy, loss) on the data
  """
    time_begin = time.time()
    score = AverageMeter()
    losses = AverageMeter()
    model.train()
    time_train = 0
    total_correct = 0

    t = tqdm(loader)
    for images, labels in t:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        total_correct += (logits.argmax(dim=1) == labels).sum()
        loss.backward()
        optimizer.step()

        acc = accuracy(logits, labels)
        n = images.size(0)
        losses.update(loss.item(), n)
        score.update(acc.item(), n)

        # t.set_description('(=> Training) Loss: {:.4f}'.format(losses.avg))

    time_train += time.time() - time_begin
    print(f"training time: {time_train}")
    return total_correct.item()/len(loader.dataset), losses.avg
