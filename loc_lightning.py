import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from src.dataloader import ActionDataset
from src.resnet import resnet18
from src.resnetv2 import resnet110
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class LitCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet110(4, num_class=2)

    def forward(self, x):
        output = self.resnet(x)
        return output

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        inputs = batch["audio"]
        labels = batch["end"].float()
        outputs = self(inputs)
        loss = F.mse_loss(outputs, labels, 1)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        # OPTIONAL

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('train_loss', avg_loss, prog_bar=True)

        return

    def validation_step(self, batch, batch_idx):
        inputs = batch["audio"]
        labels = batch["end"].float()
        outputs = self(inputs)
        loss = F.mse_loss(outputs, labels, 1)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # OPTIONAL

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        self.log('val_loss', avg_loss, prog_bar=True)

        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = lr_scheduler.MultiStepLR(optimizer, [200, 400], gamma=0.1)
        return [optimizer], [scheduler]


trainData = ActionDataset("./mask_processed.json", "./dataset/train")
dataset_sizes = {
    'train': round(0.8 * len(trainData)),
    'val': len(trainData) - round(0.8 * len(trainData))
}
[train, val] = random_split(trainData, dataset_sizes.values())

trainLoader = DataLoader(train, batch_size=8)
valLoader = DataLoader(val, batch_size=8)
dataloaders = {'train': trainLoader, 'val': valLoader}

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='lightning_logs/weights/',
    filename='action-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min',
)
early_stop_callback = EarlyStopping(
    monitor='val_sim',
    min_delta=1e-2,
    patience=20,
    mode='max',
)
trainer = pl.Trainer(gpus=2,
                     accelerator='ddp',
                     max_epochs=500,
                     callbacks=[checkpoint_callback, early_stop_callback])
model = LitCNN()

trainer.fit(model, train_dataloader=trainLoader, val_dataloaders=valLoader)
