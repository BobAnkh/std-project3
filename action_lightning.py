import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from src.dataloader import ActionDataset
from src.resnet import resnet34


class LitCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(4, num_classes=1)

    def forward(self, x):
        output = self.resnet(x)
        return output

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        inputs = batch["audio"]
        labels = batch["angle"].float()
        outputs = self(inputs)
        loss = F.mse_loss(outputs, torch.unsqueeze(labels, 1))
        sim = torch.cos(
            (outputs - torch.unsqueeze(labels, 1)) / 180 * np.pi).abs().mean()

        return {'loss': loss, 'sim': sim}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        avg_sim = torch.stack([x['sim'] for x in outputs]).mean()
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_sim', avg_sim, prog_bar=True, on_epoch=True)
        self.log('train_loss', avg_loss, prog_bar=True)

        return

    def validation_step(self, batch, batch_idx):
        inputs = batch["audio"]
        labels = batch["angle"].float()
        outputs = self(inputs)
        loss = F.mse_loss(outputs, torch.unsqueeze(labels, 1))
        sim = torch.cos(
            (outputs - torch.unsqueeze(labels, 1)) / 180 * np.pi).abs().mean()

        return {'val_loss': loss, 'sim': sim}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_sim = torch.stack([x['sim'] for x in outputs]).mean()
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_sim', avg_sim, prog_bar=True, on_epoch=True)
        self.log('val_loss', avg_loss, prog_bar=True)

        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


trainData = ActionDataset("./mask_processed.json", "./dataset/train")
dataset_sizes = {
    'train': round(0.8 * len(trainData)),
    'val': len(trainData) - round(0.8 * len(trainData))
}
[train, val] = random_split(trainData, dataset_sizes.values())

trainLoader = DataLoader(train, batch_size=8)
valLoader = DataLoader(val, batch_size=8)
dataloaders = {'train': trainLoader, 'val': valLoader}

trainer = pl.Trainer(gpus=1, max_epochs=200)
model = LitCNN()

trainer.fit(model, train_dataloader=trainLoader, val_dataloaders=valLoader)
