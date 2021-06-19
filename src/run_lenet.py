from lenet import *
from data_and_augment import *
import logging
from training_pipeline import train_fn
from evaluation import eval_fn
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def setup_training(model, args):
    """
    Setup optimiser, dataloaders, loss
    :param model
    :param args
    :return:
    config dict to run
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
    for epoch in range(args.epochs):
        # logging.info('#' * 50)
        # logging.info('Epoch [{}/{}]'.format(epoch + 1, n_epochs))
        train_score, train_loss = train_fn(model, config["optim"], config["loss"], config["data"][0], device)
        # logging.info('Train accuracy: %f', train_score)
        if epoch % 10 == 0 or epoch == (args.epochs - 1):
            val_score = eval_fn(model, config["data"][1], device)
            logging.info('Validation accuracy: %f', val_score)
            # print('Validation accuracy: %f', val_score)
            score.append({"train_loss": train_loss,
                          "train_score": train_score,
                          "val_score": val_score})
        # scheduler.step()
    return score[-1], score


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='LTH LeNet')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')

    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate 0.005')

    parser.add_argument('--dataset', type=str, default='cifar', choices=['mnist', 'cifar10'],
                        help='Data to use for training')
    # prune to 30 to get 0.1% weights
    args = parser.parse_args()
    args.dataset = 'cifar10'
    in_chan = 1 if args.dataset == 'mnist' else 3
    net = LeNet(in_channels=in_chan)
    net.apply(init_weights)
    # summary(net, (3, 32, 32),
    #         device='cuda' if torch.cuda.is_available() else 'cpu')
    metrics, train_data = run_training(net, args)
    print(f"{metrics['val_score']}")
