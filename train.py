import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from src.dataloader import ActionTrainDataset, AudioTrainDataset, VideoTrainDataset
from src.model import ActionAngle, ActionLoc, AudioClassifier, ImageClassifier


def audio_train(path):
    print('Starting training audio classification!')
    trainData = AudioTrainDataset(path)
    dataset_sizes = {
        'train': round(0.8 * len(trainData)),
        'val': len(trainData) - round(0.8 * len(trainData))
    }
    [train, val] = random_split(trainData, dataset_sizes.values())
    trainLoader = DataLoader(train, batch_size=64, num_workers=os.cpu_count())
    valLoader = DataLoader(val, batch_size=64, num_workers=os.cpu_count())
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='lightning_logs/',
        filename='audio-{epoch:02d}-{val_acc:.4f}-{train_acc:.4f}',
        save_top_k=1,
        mode='max',
    )
    trainer = pl.Trainer(gpus=2,
                         accelerator='ddp',
                         max_epochs=100,
                         callbacks=[checkpoint_callback])
    model = AudioClassifier()
    trainer.fit(model, train_dataloader=trainLoader, val_dataloaders=valLoader)
    print('Finish training. Model saved in folder lightning_logs!')


def action_angle_train(path, mask_processed_path):
    print('Starting training action angle!')
    trainData = ActionTrainDataset(mask_processed_path, path)
    dataset_sizes = {
        'train': round(0.8 * len(trainData)),
        'val': len(trainData) - round(0.8 * len(trainData))
    }
    [train, val] = random_split(trainData, dataset_sizes.values())
    trainLoader = DataLoader(train, batch_size=4, num_workers=os.cpu_count())
    valLoader = DataLoader(val, batch_size=4, num_workers=os.cpu_count())
    checkpoint_callback = ModelCheckpoint(
        monitor='val_sim',
        dirpath='lightning_logs/',
        filename='action-angle-{epoch:02d}-{val_sim:.4f}',
        save_top_k=1,
        mode='max',
    )
    trainer = pl.Trainer(gpus=2,
                         accelerator='ddp',
                         max_epochs=500,
                         callbacks=[checkpoint_callback])
    model = ActionAngle()
    trainer.fit(model, train_dataloader=trainLoader, val_dataloaders=valLoader)
    print('Finish training. Model saved in folder lightning_logs!')


def action_loc_train(path, mask_processed_path):
    print('Starting training action location!')
    trainData = ActionTrainDataset(mask_processed_path, path)
    dataset_sizes = {
        'train': round(0.8 * len(trainData)),
        'val': len(trainData) - round(0.8 * len(trainData))
    }
    [train, val] = random_split(trainData, dataset_sizes.values())
    trainLoader = DataLoader(train, batch_size=4, num_workers=os.cpu_count())
    valLoader = DataLoader(val, batch_size=4, num_workers=os.cpu_count())
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='lightning_logs/',
        filename='loc-action-{epoch:02d}-{val_loss:.4f}-{train_loss:.4f}',
        save_top_k=3,
        mode='min',
    )
    trainer = pl.Trainer(gpus=2,
                         accelerator='ddp',
                         max_epochs=500,
                         callbacks=[checkpoint_callback])
    model = ActionLoc()
    trainer.fit(model, train_dataloader=trainLoader, val_dataloaders=valLoader)
    print('Finish training. Model saved in folder lightning_logs!')


def image_train(path):
    print('Starting training image classification!')
    trainData = VideoTrainDataset(path)
    dataset_sizes = {
        'train': round(0.8 * len(trainData)),
        'val': len(trainData) - round(0.8 * len(trainData))
    }
    [train, val] = random_split(trainData, dataset_sizes.values())
    trainLoader = DataLoader(train, batch_size=128, num_workers=os.cpu_count())
    valLoader = DataLoader(val, batch_size=128, num_workers=os.cpu_count())
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='lightning_logs/',
        filename='image-{epoch:02d}-{val_acc:.4f}-{train_acc:.4f}',
        save_top_k=1,
        mode='max',
    )
    trainer = pl.Trainer(gpus=2,
                         accelerator='ddp',
                         max_epochs=50,
                         callbacks=[checkpoint_callback])
    model = ImageClassifier()
    trainer.fit(model, train_dataloader=trainLoader, val_dataloaders=valLoader)
    print('Finish training. Model saved in folder lightning_logs!')


if __name__ == '__main__':
    audio_train('./dataset/train')
    action_angle_train('./dataset/train', 'mask_processed.json')
    action_loc_train('./dataset/train', 'mask_processed.json')
    image_train('./dataset/train')
