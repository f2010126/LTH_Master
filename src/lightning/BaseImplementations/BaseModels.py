import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
import torchvision
import torchmetrics
from src.lightning.prune_model import get_masks
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


def init_weights(m):
    """
        Initialise weights acc the Xavier initialisation and bias set to 0.01
        :param m:
        :return: None
        """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# The LightningModule defines a system and not a model.
# store model, original weights
class BaseModel(pl.LightningModule):

    def __init__(self, learning_rate, loss_criterion=torch.nn.CrossEntropyLoss()):
        """
        Set up System.
        """
        super().__init__()
        # metrics
        self.accuracy = torchmetrics.Accuracy()
        self.save_hyperparameters()  # any args sent along with self get saved as hyper params.

    def forward(self, x):
        # WIll be done by subclass
        pass

    def configure_optimizers(self):
        # Put SWA here
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # per batch stuff
        data, labels = train_batch
        logits = self.forward(data)  # we already defined forward and loss in the lightning module. We'll show the
        # full code next
        train_loss = self.hparams.loss_criterion(logits, labels)
        correct = logits.argmax(dim=1).eq(labels).sum().item()
        total = len(labels)
        # Log training loss
        self.log('step_train_loss', train_loss)
        self.log('step_train_acc', self.accuracy(logits, labels))
        # loss key is  specifically named.
        return {'loss': train_loss, "correct_step": correct, "total_step": total}

    def training_epoch_end(self, outputs):
        # calculate loss for the training epoch
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = sum([x["correct_step"] for x in outputs])
        total = sum([x["total_step"] for x in outputs])
        acc = correct / total
        tensorboard_logs = {'train_loss_epoch': avg_loss, 'train_acc_epoch': acc}
        # self.log("train_epoch", tensorboard_logs,on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        logits = self.forward(data)
        val_loss = self.hparams.loss_criterion(logits, labels)
        correct = logits.argmax(dim=1).eq(labels).sum().item()
        total = len(labels)

        self.log('step_val_loss', val_loss)
        self.log('step_val_acc', self.accuracy(logits, labels))

        return {'val_loss_step': val_loss, "val_correct_step": correct, "total_step": total}

    def validation_epoch_end(self, outputs):
        # calculate loss for the validation epoch
        avg_loss = torch.stack([x['val_loss_step'] for x in outputs]).mean()
        correct = sum([x["val_correct_step"] for x in outputs])
        total = sum([x["total_step"] for x in outputs])
        acc = correct / total
        tensorboard_logs = {'val_loss_epoch': avg_loss, 'val_acc_epoch': acc}
        # self.log("val_loss_epoch", avg_loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_epoch': tensorboard_logs}

    def test_step(self, test_batch, batch_idx):
        data, labels = test_batch
        logits = self.forward(data)
        test_loss = self.hparams.loss_criterion(logits, labels)
        correct = logits.argmax(dim=1).eq(labels).sum().item()
        total = len(labels)
        self.log('step_test_loss', test_loss)
        self.log('step_test_acc', self.accuracy(logits, labels))
        return {'test_loss_step': test_loss, "test_correct_step": correct, "total_step": total}

    def test_epoch_end(self, outputs):
        # calculate loss for the test epoch
        correct = sum([x["test_correct_step"] for x in outputs])
        total = sum([x["total_step"] for x in outputs])
        acc = correct / total
        avg_loss = torch.stack([x['test_loss_step'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_loss, 'test_acc_epoch': acc}
        self.log("test_epoch", tensorboard_logs,on_step=False, on_epoch=True, prog_bar=True, logger=True)  # is written to logger

    def on_validation_end(self, *args, **kwargs) -> None:
        print("val  here")


class Net2(BaseModel):
    def __init__(self, learning_rate, loss_criterion=torch.nn.CrossEntropyLoss(), in_channels=3):
        super().__init__(learning_rate)

        self.conv1 = nn.Conv2d(in_channels, 32, (3, 3), (1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1))
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

        self.all_masks = get_masks(self, prune_amts={"linear": 0, "conv": 0, "last": 0})

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


class Resnets(BaseModel):
    def __init__(self, learning_rate, loss_criterion=torch.nn.CrossEntropyLoss(), in_channels=3):
        super().__init__(learning_rate)
        num_out_class = 10
        resnet18 = torchvision.models.resnet18(pretrained=False, progress=True)
        resnet18.fc = nn.Linear(512, num_out_class)
        # resnet18 = resnet18.to(device)
        self.model = resnet18

        self.all_masks = get_masks(self, prune_amts={"linear": 0, "conv": 0, "last": 0})

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    in_chan = 3
    loss = torch.nn.CrossEntropyLoss()
    net = Net2(learning_rate=1.2e-3, loss_criterion=loss, in_channels=in_chan)

# params in self.named_parameters()
