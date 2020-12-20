import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from src.dataloader import VideoTrainDataset
from src.resnet import resnet18


class LitCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(3)

    def forward(self, x):
        output = self.resnet(x)
        return output

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        inputs = batch["rgb"]
        labels = batch["class"]
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)

        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data) / inputs.shape[0]

        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        inputs = batch["rgb"]
        labels = batch["class"]
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data) / inputs.shape[0]

        return {'val_loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.log('acc', avg_acc, prog_bar=True, on_epoch=True)

        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


trainData = VideoTrainDataset("./dataset/train")
dataset_sizes = {
    'train': round(0.8 * len(trainData)),
    'val': len(trainData) - round(0.8 * len(trainData))
}
[train, val] = random_split(trainData, dataset_sizes.values())

trainLoader = DataLoader(train, batch_size=2)
valLoader = DataLoader(val, batch_size=2)
dataloaders = {'train': trainLoader, 'val': valLoader}

trainer = pl.Trainer(gpus=1, max_epochs=10)
model = LitCNN()

trainer.fit(model, train_dataloader=trainLoader, val_dataloaders=valLoader)
