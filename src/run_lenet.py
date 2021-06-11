from lenet import *
from data_and_augment import *
import logging
from training_pipeline import train_fn
from evaluation import eval_fn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def run_unpruned(model, n_epochs=50000, batch=60):
    model = model.to(device)
    summary(model, (1, 28, 28),
            device='cuda' if torch.cuda.is_available() else 'cpu')
    train_load, val_load, test_data = load_data(batch)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0012)
    t_max = int(len(train_load) / batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=t_max)
    logging.info('Model being trained:')
    score = []
    for epoch in range(n_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, n_epochs))
        train_score, train_loss = train_fn(model, optimizer, criterion, train_load, device)
        #logging.info('Train accuracy: %f', train_score)
        print(f"Train score: {train_score} Loss: {train_loss}")
        if epoch % 100 == 0:
            val_score = eval_fn(model, val_load, device)
            logging.info('Validation accuracy: %f', val_score)
            score.append(val_score)
        scheduler.step()


if __name__ == '__main__':
    net = LeNet()
    run_unpruned(net, n_epochs=10, batch= 60)
    print("")
