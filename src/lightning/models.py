import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch
import torchmetrics
from src.utils import init_weights
import pytorch_lightning as pl


# The LightningModule defines a system and not a model.
class Net2(pl.LightningModule):

    def __init__(self, learning_rate, in_channels=3):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()
        self.learning_rate = learning_rate

        self.conv1 = nn.Conv2d(in_channels, 32, (3, 3), (1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1))
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # per batch stuff
        print(f"train step {batch_idx}")
        data, labels = train_batch
        logits = self.forward(data)  # we already defined forward and loss in the lightning module. We'll show the full code next
        loss = self.ce(logits, labels)
        self.accuracy(logits, labels)
        correct = logits.argmax(dim=1).eq(labels).sum().item()
        total = len(labels)
        # freeze weights here
        logs = {'train_loss': loss.detach()}
        return {'loss': loss, 'log': logs, "correct": correct, "total": total}

    def validation_step(self, val_batch, batch_idx):
        print(f"val step {batch_idx}")
        data, labels = val_batch
        logits = self.forward(data)
        loss = self.ce(logits, labels)  # self.cross_entropy_loss(logits, y)
        correct = logits.argmax(dim=1).eq(labels).sum().item()
        return {'val_loss': loss, "val_correct": correct}

    def training_epoch_end(self, outputs):
        print(f" After training epoch EPOCH END: {self.accuracy.compute()}")
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        tensorboard_logs = {"Accuracy": correct / total}
        self.log('accuracy', correct / total)

    def validation_epoch_end(self, outputs):
        print(f"end of epoch validation")
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct = sum([x["val_correct"] for x in outputs])

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': correct}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def on_train_end(self):
        # prune here
        print(f"at end training. Prune here??")

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        print(f" freeze weights here? End of a Batch: {outputs}, {batch_idx}, {dataloader_idx}")



if __name__ == '__main__':
    in_chan = 3
    net = Net2(in_channels=in_chan)
    net.apply(init_weights)
    summary(net, (in_chan, 32, 32),
            device='cuda' if torch.cuda.is_available() else 'cpu')



# params in self.named_parameters()