import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split

from src.dataloader import ActionDataset
from src.resnet import resnet18
from src.resnetv2 import resnet110


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
        inputs = batch["audio"].float()
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
        inputs = batch["audio"].float()
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
    'train': len(trainData)
}

trainLoader = DataLoader(trainData, batch_size=4, num_workers=os.cpu_count())


trainer = pl.Trainer(gpus=1,
                     max_epochs=100)
model = LitCNN().load_from_checkpoint('lightning_logs/version_1/action-epoch=300-val_loss=613.8790.ckpt')
model.cuda()
model.eval()
# trainer.test(model, test_dataloaders=trainLoader)

out_rel = []

for sample in trainLoader:
    inputs = sample["audio"].float().cuda()
    outputs = model(inputs)
    tmp = list(map(lambda i:
            {"class": int(sample["class"][i]),
             "label": sample["label"][i],
             "end": sample["end"][i].detach().cpu().numpy().tolist(),
             "outputs": outputs[i].detach().cpu().numpy().tolist()
             }, range(len(sample['class']))))
    out_rel.append(tmp)

json.dump(out_rel, open('loc_train_all_results.json', 'w', encoding='utf-8'), ensure_ascii=False)