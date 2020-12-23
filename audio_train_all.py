import json
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split

from src.dataloader import AudioTrainDataset
from src.resnet import resnet18, resnet34
from src.resnetv2 import resnet20


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
    'train': len(trainData)
}
trainLoader = DataLoader(trainData, batch_size=32, num_workers=os.cpu_count())


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
trainer = pl.Trainer(gpus=1,
                     max_epochs=100)
model = LitCNN().load_from_checkpoint('lightning_logs/version_4/audio-epoch=56-val_acc=0.9922.ckpt')
model.cuda()
model.eval()
# trainer.test(model, test_dataloaders=trainLoader)

out_rel = []
for sample in trainLoader:
    inputs = sample["audio"].float().cuda()
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    tmp = list(map(lambda i:
            {"class": sample["class"][i].detach().cpu().numpy().tolist(),
             "label": sample["label"][i],
             "outputs": outputs[i].detach().cpu().numpy().tolist(),
             "pred": preds[i].detach().cpu().numpy().tolist()}, range(len(sample))))
    out_rel.append(tmp)

json.dump(out_rel, open('train_all_results.json', 'w', encoding='utf-8'), ensure_ascii=False)
