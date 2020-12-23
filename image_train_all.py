import json
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import src.resnetv2 as resnetv2
from src.dataloader import VideoTrainDataset
from src.resnet import resnet18


class LitCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnetv2.resnet44(3)

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

        return {'loss': loss, 'train_acc': acc}

    def training_epoch_end(self, outputs):
        # OPTIONAL

        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()

        self.log('train_acc', avg_acc, prog_bar=True)

        return

    def validation_step(self, batch, batch_idx):
        inputs = batch["rgb"]
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
        scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 40], gamma=0.1)
        return [optimizer], [scheduler]


trainData = VideoTrainDataset("./dataset/train")
dataset_sizes = {
    'train': len(trainData)
}

trainLoader = DataLoader(trainData, batch_size=64, num_workers=os.cpu_count())


trainer = pl.Trainer(gpus=1,
                     max_epochs=100)
model = LitCNN().load_from_checkpoint('weights/image-resnet44.ckpt')
model.cuda()
model.eval()
# trainer.test(model, test_dataloaders=trainLoader)

out_rel = []

for sample in trainLoader:
    inputs = sample["rgb"].float().cuda()
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    tmp = list(map(lambda i:
            {"class": sample["class"][i].detach().cpu().numpy().tolist(),
             "label": sample["label"][i],
             "pred": preds[i].detach().cpu().numpy().tolist()
             }, range(len(sample['class']))))
    out_rel.append(tmp)

json.dump(out_rel, open('image_train_all_results.json', 'w', encoding='utf-8'), ensure_ascii=False)
