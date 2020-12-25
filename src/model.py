import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    '''1x1 convolution'''
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv1x1(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class ResNet(nn.Module):
    def __init__(self, input_channel, block, layers, num_classes=10, bn=64):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(input_channel,
                               16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn3 = nn.BatchNorm2d(bn)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet20(input_channel, num_class=10):
    model = ResNet(input_channel, BasicBlock, [3, 3, 3, 3], num_class)
    return model


def resnet32(input_channel, num_class=10):
    model = ResNet(input_channel, BasicBlock, [5, 5, 5, 5], num_class)
    return model


def resnet44(input_channel, num_class=10):
    model = ResNet(input_channel, Bottleneck, [7, 7, 7, 7], num_class, bn=256)
    return model


def resnet110(input_channel, num_class=10):
    model = ResNet(input_channel, Bottleneck, [18, 18, 18, 18], num_class, bn=256)
    return model


def resnet302(input_channel, num_class=10):
    model = ResNet(input_channel, Bottleneck, [50, 50, 50, 50], num_class, bn=256)
    return model


class AudioClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet20(4)

    def forward(self, x):
        output = self.resnet(x)
        return output

    def training_step(self, batch, batch_idx):
        inputs = batch["audio"].float()
        labels = batch["class"]
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data) / inputs.shape[0]
        return {'loss': loss, 'train_acc': acc}

    def training_epoch_end(self, outputs):
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
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('val_acc', avg_acc, prog_bar=True, on_epoch=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = lr_scheduler.MultiStepLR(optimizer, [25, 50, 70], gamma=0.1)
        return [optimizer], [scheduler]


class ActionAngle(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet110(4, num_class=1)

    def forward(self, x):
        output = self.resnet(x)
        return output

    def training_step(self, batch, batch_idx):
        inputs = batch["audio"].float()
        labels = batch["angle"].float()
        outputs = self(inputs)
        loss = F.mse_loss(outputs, torch.unsqueeze(labels, 1))
        sim = torch.cos(
            (outputs - torch.unsqueeze(labels, 1)) / 180 * np.pi).abs().mean()
        return {'loss': loss, 'sim': sim}

    def training_epoch_end(self, outputs):
        avg_sim = torch.stack([x['sim'] for x in outputs]).mean()
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_sim', avg_sim, prog_bar=True, on_epoch=True)
        self.log('train_loss', avg_loss, prog_bar=True)
        return

    def validation_step(self, batch, batch_idx):
        inputs = batch["audio"].float()
        labels = batch["angle"].float()
        outputs = self(inputs)
        loss = F.mse_loss(outputs, torch.unsqueeze(labels, 1))
        sim = torch.cos(
            (outputs - torch.unsqueeze(labels, 1)) / 180 * np.pi).abs().mean()
        return {'val_loss': loss, 'sim': sim}

    def validation_epoch_end(self, outputs):
        avg_sim = torch.stack([x['sim'] for x in outputs]).mean()
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_sim', avg_sim, prog_bar=True, on_epoch=True)
        self.log('val_loss', avg_loss, prog_bar=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = lr_scheduler.MultiStepLR(optimizer, [200, 400], gamma=0.1)
        return [optimizer], [scheduler]


class ActionLoc(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet110(4, num_class=2)

    def forward(self, x):
        output = self.resnet(x)
        return output

    def training_step(self, batch, batch_idx):
        inputs = batch["audio"].float()
        labels = batch["end"].float()
        outputs = self(inputs)
        loss = F.mse_loss(outputs, labels, 1)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
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
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = lr_scheduler.MultiStepLR(optimizer, [200, 400], gamma=0.1)
        return [optimizer], [scheduler]


class ImageClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet44(3)

    def forward(self, x):
        output = self.resnet(x)
        return output

    def training_step(self, batch, batch_idx):
        inputs = batch["rgb"]
        labels = batch["class"]
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data) / inputs.shape[0]
        return {'loss': loss, 'train_acc': acc}

    def training_epoch_end(self, outputs):
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
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('val_acc', avg_acc, prog_bar=True, on_epoch=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 40], gamma=0.1)
        return [optimizer], [scheduler]
