import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader

from src.dataloader import trainDataset
from src.model import AudioEmbed


def train_task1(root_path, num_epochs=50):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task1/test/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': 1, ‘audio_0001’: 3, ...}
    class number:
        ‘061_foam_brick’: 0
        'green_basketball': 1
        'salt_cylinder': 2
        'shiny_toy_gun': 3
        'stanley_screwdriver': 4
        'strawberry': 5
        'toothpaste_box': 6
        'toy_elephant': 7
        'whiteboard_spray': 8
        'yellow_block': 9
    '''
    trainData = trainDataset(root_path, True, False)

    dataset_sizes = {'train': round(
        0.8 * len(trainData)), 'val': len(trainData) - round(0.8 * len(trainData))}
    [train, val] = torch.utils.data.random_split(
        trainData, dataset_sizes.values())

    trainLoader = DataLoader(train, batch_size=4)
    valLoader = DataLoader(val, batch_size=4)
    dataloaders = {'train': trainLoader, 'val': valLoader}

    mod = AudioEmbed()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    mod.to(device)

    optimizer = optim.SGD(mod.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    since = time.time()
    # for batch_ndx, sample in enumerate(trainLoader):
    #     # print(batch_ndx, sample, sample["audio"].shape, sample["rgb"].shape)
    #     print(mod(sample["audio"].float()).shape)
    #     break
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                mod.train()  # Set model to training mode
            else:
                mod.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for sample in dataloaders[phase]:
                inputs = sample["audio"].float().to(device)
                labels = sample["class"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = mod(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            time_epoch = time.time() - since
            print('Training in {:.0f}m {:.0f}s'.format(
                time_epoch // 60, time_epoch % 60))


if __name__ == "__main__":
    train_task1("./dataset/train")
