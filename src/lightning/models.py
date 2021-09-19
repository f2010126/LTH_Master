import torch.nn as nn
import torch.nn.functional as F
import copy
import torch

from prune_model import get_masks
import pytorch_lightning as pl


def count_rem_weights(model):
    """
    Percetage of weights that remain for training
    :param model:
    :return: % of weights remaining
    """
    total_weights = 0
    rem_weights = 0
    for name, module in model.named_modules():
        if any([isinstance(module, cl) for cl in [torch.nn.Conv2d, torch.nn.Linear]]):
            rem_weights += torch.count_nonzero(module.weight)
            total_weights += sum([param.numel() for param in module.parameters()])
    # return % of non 0 weights
    return rem_weights.item() / total_weights * 100


# The LightningModule defines a system and not a model.
class Net2(pl.LightningModule):

    def init_weights(self, m):
        """
        Initialise weights acc the Xavier initialisation and bias set to 0.01
        :param m:
        :return: None
        """
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, learning_rate, loss_criterion=torch.nn.CrossEntropyLoss(),in_channels=3):
        """
        Set up System.
        """
        super().__init__()
        self.save_hyperparameters()  # any args sent along with self get saved as hyper params.

        self.conv1 = nn.Conv2d(in_channels, 32, (3, 3), (1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1))
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.apply(self.init_weights)

        self.all_masks = {key: mask for key, mask in get_masks(self, prune_amts={"linear": 0, "conv": 0, "last": 0})}
        self.original_state_dict = copy.deepcopy(self.state_dict())


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # per batch stuff
        data, labels = train_batch
        logits = self.forward(data)  # we already defined forward and loss in the lightning module. We'll show the
        # full code next
        loss = self.hparams.loss_criterion(logits, labels)
        correct = logits.argmax(dim=1).eq(labels).sum().item()
        total = len(labels)
        self.log('train_loss', loss.detach())
        logs = {'train_loss': loss.detach()}
        return {'loss': loss, 'log': logs, "correct": correct, "total": total}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        acc = correct / total
        tensorboard_logs = {'train_loss': avg_loss, 'train_acc': acc}
        print(f"train Epoch end acc {tensorboard_logs}")
        self.log("train_epoch", tensorboard_logs)

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        logits = self.forward(data)
        loss = self.hparams.loss_criterion(logits, labels)
        correct = logits.argmax(dim=1).eq(labels).sum().item()
        total = len(labels)
        return {'val_loss': loss, "val_correct": correct, "total": total}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct = sum([x["val_correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        acc = correct / total

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        self.log("validation_epoch", tensorboard_logs)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, test_batch, batch_idx):
        data, labels = test_batch
        logits = self.forward(data)
        loss = self.hparams.loss_criterion(logits, labels)
        correct = logits.argmax(dim=1).eq(labels).sum().item()
        total = len(labels)
        return {'test_loss': loss, "test_correct": correct, "total": total}

    def test_epoch_end(self, outputs):
        correct = sum([x["test_correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        acc = correct / total
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': acc}
        print(f"test Epoch end acc {tensorboard_logs}")
        self.log("test_epoch", tensorboard_logs) # is written to logger

    def on_test_end(self):
        print("End")


if __name__ == '__main__':
    in_chan = 3
    loss = torch.nn.CrossEntropyLoss()
    net = Net2(learning_rate=1.2e-3, loss_criterion=loss, in_channels=in_chan)

# params in self.named_parameters()
