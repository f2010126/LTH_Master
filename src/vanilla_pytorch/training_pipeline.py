from tqdm import tqdm
import time
from .evaluation import AverageMeter, accuracy


def train_fn(model, optimizer, criterion, loader, device, epoch,
             swa_start=0, swa_model=None, scheduler=None, swa_scheduler=None):
    """
  Training method
  :param model: model to train
  :param optimizer: optimization algorithm
  :param criterion: loss function
  :param loader: data loader for either training or testing set
  :param device: torch device
  :param epoch: needed for swa
  :param swa: boolean to indicate if SWA used
  :param swa_model: for when swa is true
  :return: (accuracy, loss) on the data
  """
    time_begin = time.time()

    score = AverageMeter()
    losses = AverageMeter()
    model.train()
    time_train = 0
    total_correct = 0

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, (images, labels) in pbar:
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

    if epoch > swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

    time_train += time.time() - time_begin
    return total_correct / len(loader.dataset), losses.avg
