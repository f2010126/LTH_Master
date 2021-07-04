from tqdm import tqdm
import time
from evaluation import AverageMeter, accuracy


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
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        loss.backward()
        # freeze pruned weights by making their gradients 0
        for module in model.modules():
            if hasattr(module, "weight_mask"):
                weight = next(param for name, param in module.named_parameters() if "weight" in name)
                weight.grad = weight.grad * module.weight_mask
                # tensor = param.data.cpu().numpy()
                # grad_tensor = param.grad.data.cpu().numpy()
                # # set grad to 0 for 0 tensors, ie freeze their training
                # grad_tensor = np.where(tensor == 0, 0, grad_tensor)
                # param.grad.data = torch.from_numpy(grad_tensor).to(device)

        optimizer.step()

        acc = accuracy(logits.detach(), labels)
        n = images.shape[0]
        losses.update(loss.item(), n)
        score.update(acc.item(), n)

    time_train += time.time() - time_begin
    return total_correct / len(loader.dataset), losses.avg
