import logging
import argparse
import time
import torch
from torchsummary import summary
from torch.optim.swa_utils import AveragedModel
from src.vanilla_pytorch.data_and_augment import load_cifar10_data, load_mnist_data
from src.vanilla_pytorch.training_pipeline import train_fn
from src.vanilla_pytorch.evaluation import eval_fn
from src.vanilla_pytorch.EarlyStopping import Py_EarlyStop
from src.vanilla_pytorch.utils import init_weights
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.vanilla_pytorch.models.convnets import Net2
from src.vanilla_pytorch.models.resnets import Resnets


def setup_training(model, device, args):
    """
    Setup optimiser, dataloaders, loss
    :param model
    :param args
    :param device cpu or gpu
    :return: config dict to run
    """
    if args.dataset == 'mnist':
        train_load, val_load, test_data = load_mnist_data(args.batch_size)
    else:
        train_load, val_load, test_data = load_cifar10_data(args.batch_size)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    # TODO: optimiser? Scheduler?
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    max_epochs = args.epochs  # if (args.epochs > math.floor(args.iterations / len(train_load))) else math.floor(
    # args.iterations / len(train_load))
    return {"optim": optimizer,
            "data": (train_load, val_load, test_data),
            "loss": criterion,
            "max_epochs": max_epochs}


def run_training(model, device, args=None):
    model = model.to(device)
    config = setup_training(model, device, args)
    logging.info('Model being trained:')
    # init an SWA Model?
    swa_model = AveragedModel(model)
    scheduler = CosineAnnealingLR(config["optim"], T_max=100)
    swa_start = 5
    swa_scheduler = SWALR(config["optim"], swa_lr=0.05)


    score = []
    e_stop = Py_EarlyStop(patience=10, verbose=True)

    for epoch in range(config["max_epochs"]):
        logging.info('Epoch [{}/{}]'.format(epoch + 1, config["max_epochs"]))
        train_score, train_loss = train_fn(model, config["optim"], config["loss"], config["data"][0], device,
                                           epoch=epoch, swa_start=swa_start, swa_model=swa_model,
                                           scheduler=scheduler, swa_scheduler=swa_scheduler)

        torch.optim.swa_utils.update_bn(config["data"][0], swa_model,device=device)
        val_score, val_loss = eval_fn(swa_model, config["data"][1], device, config["loss"])
        print('Validation accuracy: %f', val_score)
        score.append({"epoch": epoch,
                      "train_loss": train_loss,
                      "train_score": train_score,
                      "val_score": val_score,
                      "val_loss": val_loss})

        if args.early_stop:
            e_stop(val_loss, model)
            if e_stop.early_stop:
                print("STOP")
                stop_epoch = epoch
                break


    test_score, test_loss = eval_fn(swa_model, config["data"][2], device, config["loss"])
    stop_epoch = sorted(score, key=lambda k: k['val_loss'])[0]['epoch']
    print(f" Evaluating on Test: Loss {test_loss} and score {test_score}")
    return score[-1], stop_epoch, score


if __name__ == '__main__':
    start = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description='LTH Model')
    parser.add_argument('--model', type=str, default='Net2',
                        help='Class name of model to train',
                        choices=['LeNet', 'Net2', 'LeNet300', 'Resnets'])
    parser.add_argument('--batch-size', type=int, default=60,
                        help='input batch size for training (default: 60)')

    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--iterations', type=int, default=50000,
                        help='number of iterations to train (default: 50000)')

    parser.add_argument('--lr', type=float, default=1.2e-3,
                        help='learning rate 1.2e-3')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'],
                        help='Data to use for training')
    parser.add_argument('--early-stop',
                        action='store_true', help='Does Early if enabled')
    parser.add_argument('--early-delta', type=float, default=.009, help='Difference b/w best and current to decide to '
                                                                        'stop early')
    parser.add_argument('--use-swa',
                        action='store_true', help='Uses SWA if enabled')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = parser.parse_args()

    in_chan, img = (1, 28) if args.dataset == 'mnist' else (3, 32)
    net = eval(args.model)(in_channels=in_chan)
    net.apply(init_weights)
    print(f"Arguments: {args}")
    summary(net, (in_chan, img, img),
            device=device.type)
    metrics, es_epoch, _ = run_training(net, device, args)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    print(f"Validation: {metrics['val_score']} and Stopping :{es_epoch}")

# LeNet300- 50kitr/60batch Adam 1.2e-3
# Conv2 20k itr/60 batch Adam 2e-4
# Resnet 30k/128batch SGD .1,.01,.001 momentum 09 20% conv
