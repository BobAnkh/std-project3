import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from src.dataloader import AudioTrainDataset
from src.resnet import resnet18, resnet34
from src.resnetv2 import resnet20
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class LitCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet20(4)

    def forward(self, x):
        output = self.resnet(x)
        return output

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        inputs = batch["audio"].float()
        labels = batch["class"]
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)

        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data) / inputs.shape[0]

        return {'loss': loss, 'train_acc': acc}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        self.log('train_acc', avg_acc, prog_bar=True, on_epoch=True)

        return

    def validation_step(self, batch, batch_idx):
        inputs = batch["audio"].float()
        labels = batch["class"]
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data) / inputs.shape[0]

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('val_acc', avg_acc, prog_bar=True, on_epoch=True)

        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = lr_scheduler.MultiStepLR(optimizer, [25, 50, 70], gamma=0.1)
        return [optimizer], [scheduler]


trainData = AudioTrainDataset("./dataset/train")
dataset_sizes = {
    'train': round(0.8 * len(trainData)),
    'val': len(trainData) - round(0.8 * len(trainData))
}
[train, val] = random_split(trainData, dataset_sizes.values())

trainLoader = DataLoader(train, batch_size=64, num_workers=os.cpu_count())
valLoader = DataLoader(val, batch_size=64, num_workers=os.cpu_count())
dataloaders = {'train': trainLoader, 'val': valLoader}

checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath='lightning_logs/weights/',
    filename='audio-{epoch:02d}-{val_acc:.4f}-{train_acc:.4f}',
    save_top_k=1,
    mode='max',
)
# early_stop_callback = EarlyStopping(
#     monitor='val_acc',
#     min_delta=1e-3,
#     patience=20,
#     mode='max',
# )
trainer = pl.Trainer(gpus=2,
                     accelerator='ddp',
                     max_epochs=100,
                     callbacks=[checkpoint_callback])
model = LitCNN()

trainer.fit(model, train_dataloader=trainLoader, val_dataloaders=valLoader)
