from convnets import *
from data_and_augment import *
import logging
from training_pipeline import train_fn
from evaluation import eval_fn
import argparse
import time
from lenet import *
from convnets import *
from EarlyStopping import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def setup_training(model, args):
    """
    Setup optimiser, dataloaders, loss
    :param model
    :param args
    :return: config dict to run
    """
    if args.dataset == 'mnist':
        train_load, val_load, test_data = load_mnist_data(args.batch_size)
    else:
        train_load, val_load, test_data = load_cifar10_data(args.batch_size)

        # result = (on_false, on_true)[condition]
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # TODO: optimiser? Scheduler?
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.12)
    # t_max = int(len(train_load) / batch)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    return {"optim": optimizer,
            "data": (train_load, val_load, test_data),
            "loss": criterion}


def run_training(model, args=None):
    model = model.to(device)
    config = setup_training(model, args)
    logging.info('Model being trained:')
    score = []
    stop_epoch = args.epochs
    e_stop = EarlyStopping(min_delta=args.early_delta)
    for epoch in range(args.epochs):
        # logging.info('Epoch [{}/{}]'.format(epoch + 1, n_epochs))
        train_score, train_loss = train_fn(model, config["optim"], config["loss"], config["data"][0], device)
        val_score, val_loss = eval_fn(model, config["data"][1], device, config["loss"])
        # logging.info('Validation accuracy: %f', val_score)
        score.append({"epoch": epoch,
                      "train_loss": train_loss,
                      "train_score": train_score,
                      "val_score": val_score,
                      "val_loss": val_loss})
        if args.early_stop:
            e_stop(val_loss)
            if e_stop.early_stop:
                stop_epoch = epoch
                break
        # if epoch % 2 == 0 or epoch == (args.epochs - 1):
        #     val_score, val_loss = eval_fn(model, config["data"][1], device, config["loss"])
        #     # logging.info('Validation accuracy: %f', val_score)
        #     # print(f"Validation loss {val_loss} and training loss {train_loss} best loss {e_stop.best_loss}")
        #     score.append({"train_loss": train_loss,
        #                   "train_score": train_score,
        #                   "val_score": val_score,
        #                   "val_loss": val_loss})
        #     if args.early_stop:
        #         e_stop(val_loss)
        #         if e_stop.early_stop:
        #             stop_epoch = epoch
        #             break

        # scheduler.step()
    return score[-1], stop_epoch, score


if __name__ == '__main__':
    start = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description='LTH Model')
    parser.add_argument('--model', type=str, default='Net2',
                        help='Class name of modeto train',
                        choices=['LeNet', 'Net2'])
    parser.add_argument('--batch-size', type=int, default=512,
                        help='input batch size for training (default: 128)')

    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--iterations', type=int, default=1700,
                        help='number of iterations to train (default: 1700)')

    parser.add_argument('--lr', type=float, default=0.0012,
                        help='learning rate 0.0012')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'],
                        help='Data to use for training')
    parser.add_argument('--early-stop', type=bool, default=False, help='Should Early stopping be done? Default False')
    parser.add_argument('--early-delta', type=float, default=.009, help='Difference b/w best and current to decide to stop early')
    args = parser.parse_args()
    in_chan, img = (1, 32) if args.dataset == 'mnist' else (3, 32)
    net = eval(args.model)(in_channels=in_chan)
    net.apply(init_weights)
    summary(net, (in_chan, img, img),
            device='cuda' if torch.cuda.is_available() else 'cpu')
    metrics, es_epoch, _ = run_training(net, args)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    print(f"Validation: {metrics['val_score']} and Stopping :{0 if not args.early_stop else es_epoch}")
