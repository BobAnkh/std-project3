import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from src.dataloader import AudioTrainDataset
from src.resnet import resnet34


class LitCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(4, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 8, 5)
        # self.fc1 = nn.Linear(8 * 122 * 166, 10000)
        # self.fc2 = nn.Linear(10000, 2000)
        # self.fc3 = nn.Linear(2000, 200)
        # self.fc4 = nn.Linear(200, 10)
        self.resnet = resnet34()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
        output = self.resnet(x)
        return output

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        inputs = batch["audio"]
        labels = batch["class"]
        outputs = self(inputs)
        loss = F.nll_loss(outputs, labels)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["audio"]
        labels = batch["class"]
        outputs = self(inputs)
        loss = F.nll_loss(outputs, labels)
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data) / inputs.shape[0]

        return {'val_loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.log('acc', avg_acc)
        print(avg_acc)
        return {'acc': avg_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


trainData = AudioTrainDataset("./dataset/train")
dataset_sizes = {
    'train': round(0.8 * len(trainData)),
    'val': len(trainData) - round(0.8 * len(trainData))
}
[train, val] = torch.utils.data.random_split(trainData, dataset_sizes.values())

trainLoader = DataLoader(train, batch_size=2)
valLoader = DataLoader(val, batch_size=2)
dataloaders = {'train': trainLoader, 'val': valLoader}

trainer = pl.Trainer(gpus=1, max_epochs=10)
model = LitCNN()

trainer.fit(model, train_dataloader=trainLoader, val_dataloaders=valLoader)
