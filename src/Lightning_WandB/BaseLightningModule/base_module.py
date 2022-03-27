import torch
import os
import torchvision
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
from torchsummary import summary
import copy
from torch.nn.utils.prune import is_pruned
import torch.nn.init as init
import wandb

try:
    from .ResnetModel import ResNet18
    from .DropoutResnet import ResNet18_Dropout
    from src.Lightning_WandB.utils import apply_prune, plot_grad_flow, count_rem_weights
except ImportError:
    from src.Lightning_WandB.BaseLightningModule.ResnetModel import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    from src.Lightning_WandB.BaseLightningModule.DropoutResnet import ResNet18_Dropout
    from src.Lightning_WandB.utils import apply_prune, plot_grad_flow, count_rem_weights


def create_model(arch_type):
    return {'resnet18': ResNet18(low_dim=10),
            'resnet18_dropout': ResNet18_Dropout(low_dim=10),
            'torch_resnet': torchvision_renet()}[arch_type]


# Gaussian Glorot initialization
def init_weights(m):
    """
        Initialise weights acc the Xavier initialisation and bias set to 0.01
        :param m:
        :return: None
        """
    if isinstance(m, torch.nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def weight_reinit(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)


def torchvision_renet():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


class LitSystem94Base(LightningModule):
    def __init__(self, batch_size, arch, experiment_dir='experiments', reset_itr=0, prune_amount=0.2, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model(arch)
        self.model.apply(init_weights)

    def show_model_summary(self):
        # for Cifar10 now.
        summary(self.model, (3, 32, 32))

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def on_fit_start(self):
        print(f"LOCAL RANK {self.local_rank}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000  # // self.hparams.batch_size
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


class LitSystemPrune(LightningModule):
    def __init__(self, batch_size, arch, experiment_dir='experiments',
                 reset_itr=0, prune_amount=0.2, lr=0.05, weight_decay=0.0001):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model(arch)
        self.model.apply(init_weights)
        self.final_wgts = None
        # init the masks in the model
        apply_prune(self, 0.0, "magnitude", True)
        self.original_wgts = copy.deepcopy(self.state_dict())  # maintain the weights
        self.prepare_data_per_node = False

    def on_train_start(self):
        weight_prune = count_rem_weights(self)
        self.log('model_weight', weight_prune, on_epoch=True, logger=True, sync_dist=True)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        # # If module has reset itr is same as current itr, update the weights dict
        if (self.global_step == self.hparams.reset_itr):
            self.original_wgts = copy.deepcopy(self.state_dict())

        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc * 100, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_acc", acc * 100, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        # return the SGD used
        optimizer = torch.optim.SGD(params=self.model.parameters(),
                                    lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay)
        # torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    # PRUNING FUNCTIONS #

    def reset_weights(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and (
                    f"{name}.weight_orig" in self.original_wgts.keys()):
                # do nothing for unpruned weights?
                if is_pruned(module) is False:
                    continue
                with torch.no_grad():
                    module.weight_orig.copy_(self.original_wgts[f'{name}.weight_orig'])
                    module.bias.copy_(self.original_wgts[f'{name}.bias'])

    def on_after_backward(self):
        # freeze pruned weights by making their gradients 0. using the Mask.
        # for module in self.modules():
        #     if hasattr(module, "weight_mask"):
        #         weight = next(param for name, param in module.named_parameters() if "weight" in name)
        #         weight.grad = weight.grad * module.weight_mask

        # Alternate way of freezing.
        EPS = 1e-6
        for name, p in self.named_parameters():
            if 'weight' in name:
                tensor = p.data
                grad_tensor = p.grad
                grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                p.grad.data = grad_tensor


class LitSystemRandom(LightningModule):
    def __init__(self, batch_size, arch, experiment_dir='experiments',
                 prune_amount=0.2, lr=0.05, weight_decay=0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(arch)
        self.model.apply(weight_reinit)
        # init the masks in the model
        apply_prune(self, 0.0, "random", True)
        self.prepare_data_per_node = False

    def random_init_weights(self):
        self.model.apply(weight_reinit)

    def on_train_start(self):
        weight_prune = count_rem_weights(self)
        self.log('model_weight', weight_prune, on_epoch=True, logger=True, sync_dist=True)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc * 100, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_acc", acc * 100, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def on_after_backward(self):
        # freeze weights by making their gradients 0. using the Mask.
        for module in self.modules():
            if hasattr(module, "weight_mask"):
                weight = next(param for name, param in module.named_parameters() if "weight" in name)
                weight.grad = weight.grad * module.weight_mask

    def configure_optimizers(self):
        # return the SGD used
        optimizer = torch.optim.SGD(params=self.model.parameters(),
                                    lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay)
        # torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer


class LitSystemSSLPrune(LightningModule):

    def create_model(self):
        model = ResNet18(low_dim=10)
        # init the fc layer
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        pretrain_path = 'Pretrain_SSL_Model_best.pth'  # 'Pretrain_SSL_Model/Pretrain_SSL_Model_best.pth'
        if os.path.isfile(pretrain_path):
            print("=> loading checkpoint '{}'".format(pretrain_path))
            checkpoint = torch.load(pretrain_path)
            state_dict = checkpoint['state_dict']

            new_state_dict = dict()
            for old_key, value in state_dict.items():
                if old_key.startswith('backbone') and 'fc' not in old_key:
                    new_key = old_key.replace('backbone.', '')
                    new_state_dict[new_key] = value

            msg = model.load_state_dict(new_state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(pretrain_path))
        else:
            print("=> no checkpoint found at '{}'".format(pretrain_path))

        return model

    def __init__(self, batch_size, arch, experiment_dir='experiments',
                 reset_itr=0, prune_amount=0.2, lr=0.05, weight_decay=0.0001):
        super().__init__()

        self.save_hyperparameters()
        self.model = self.create_model()
        # init the masks in the model
        # apply_prune(self, 0.0, "magnitude", True)
        self.original_wgts = copy.deepcopy(self.state_dict())  # maintain the weights
        self.prepare_data_per_node = False

    def on_train_start(self):
        weight_prune = count_rem_weights(self)
        self.log('model_weight', weight_prune, on_epoch=True, logger=True, sync_dist=True)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        # # If module has reset itr is same as current itr, update the weights dict
        if (self.global_step == self.hparams.reset_itr):
            self.original_wgts = copy.deepcopy(self.state_dict())

        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc * 100, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_acc", acc * 100, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        # return the SGD used
        optimizer = torch.optim.SGD(params=self.model.parameters(),
                                    lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay)
        # torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def reset_weights(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and (
                    f"{name}.weight_orig" in self.original_wgts.keys()):
                # do nothing for unpruned weights?
                if is_pruned(module) is False:
                    continue
                with torch.no_grad():
                    module.weight_orig.copy_(self.original_wgts[f'{name}.weight_orig'])
                    module.bias.copy_(self.original_wgts[f'{name}.bias'])

    def on_after_backward(self):
        # freeze pruned weights by making their gradients 0. using the Mask.
        EPS = 1e-6
        for name, p in self.named_parameters():
            if 'weight' in name:
                tensor = p.data
                grad_tensor = p.grad
                grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                p.grad.data = grad_tensor

        # for module in self.modules():
        #     if hasattr(module, "weight_mask"):
        #         weight = next(param for name, param in module.named_parameters() if "weight" in name)
        #         weight.grad = weight.grad * module.weight_mask
