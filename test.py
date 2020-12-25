import json
import os

import torch
from torch.utils.data.dataloader import DataLoader

from src.dataloader import AudioTestDataset
from src.model import AudioClassifier
from src.audio_process import pre_process


def test_task1(root_path):
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
    print('Pre-processing audio data!')
    pre_process(root_path)
    print('Starting test!')
    test_data = AudioTestDataset(root_path)
    test_loader = DataLoader(test_data,
                             batch_size=1,
                             num_workers=os.cpu_count())
    model = AudioClassifier().load_from_checkpoint('weights/audio-resnet20.ckpt')
    model.cuda()
    model.eval()

    results = {}
    for sample in test_loader:
        inputs = sample["audio"].float().cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        mapper = map(
            lambda i:
            (sample['label'][i], preds[i].detach().cpu().numpy().tolist()),
            range(len(sample['label'])))
        results.update(mapper)

    return results


def test_task2(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task2/test/0/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': 23, ‘audio_0001’: 11, ...}
    This means audio 'audio_0000.pkl' is matched to video 'video_0023' and ‘audio_0001’ is matched to 'video_0011'.
    '''
    print('Pre-processing audio data!')
    pre_process(root_path)
    results = None
    return results


def test_task3(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task3/test/0/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': -1, ‘audio_0001’: 12, ...}
    This means audio 'audio_0000.pkl' is not matched to any video and ‘audio_0001’ is matched to 'video_0012'.
    '''
    print('Pre-processing audio data!')
    pre_process(root_path)
    results = None
    return results


if __name__ == "__main__":
    task1 = test_task1("./dataset/task1/test")
    json.dump(task1,
              open('task1_results.json', 'w', encoding='utf-8'),
              ensure_ascii=False)
