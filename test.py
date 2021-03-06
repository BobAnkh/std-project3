import json
import os

import pandas as pd

import torch
from torch.utils.data.dataloader import DataLoader

from src.dataloader import AudioTestDataset, VideoTestDataset, ActionTestDataset
from src.model import AudioClassifier, ImageClassifier, ActionAngle, ActionLoc
from src.audio_process import pre_process
from src.image_center import test_mask_process
from src.KM import find_match


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
    print('----------Start task1 test----------')
    print('Pre-processing audio data!')
    pre_process(root_path)
    print('Starting test!')
    test_data = AudioTestDataset(root_path)
    test_loader = DataLoader(test_data, batch_size=1)
    model = AudioClassifier().load_from_checkpoint(
        'weights/audio-resnet20.ckpt')
    model.cuda()
    model.eval()

    results = {}
    for sample in test_loader:
        inputs = sample['audio'].float().cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        mapper = map(
            lambda i:
            (sample['label'][i], preds[i].detach().cpu().numpy().tolist()),
            range(len(sample['label'])))
        results.update(mapper)
    print('----------Finish task1 test----------')
    return results


def test_task2(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task2/test/0/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': 23, ‘audio_0001’: 11, ...}
    This means audio 'audio_0000.pkl' is matched to video 'video_0023' and ‘audio_0001’ is matched to 'video_0011'.
    '''
    print('----------Start task2 test----------')
    print('Pre-processing audio data!')
    pre_process(root_path)

    print('Processing mask data...')
    mask_list = test_mask_process(root_path)

    print('Image classification...')
    image_rel = []
    test_data1 = VideoTestDataset(root_path)
    test_loader1 = DataLoader(test_data1, batch_size=16)
    model1 = ImageClassifier().load_from_checkpoint(
        'weights/image-resnet44.ckpt')
    model1.cuda()
    model1.eval()
    for sample in test_loader1:
        inputs = sample['rgb'].float().cuda()
        outputs = model1(inputs)
        _, preds = torch.max(outputs, 1)
        tmp = list(
            map(
                lambda i: {
                    'label': sample['label'][i],
                    'class': preds[i].detach().cpu().numpy().tolist()
                }, range(len(sample['label']))))
        image_rel.append(tmp)

    print('Audio classification...')
    audio_rel = []
    test_data2 = AudioTestDataset(root_path)
    test_loader2 = DataLoader(test_data2, batch_size=1)
    model2 = AudioClassifier().load_from_checkpoint(
        'weights/audio-resnet20.ckpt')
    model2.cuda()
    model2.eval()
    for sample in test_loader2:
        inputs = sample['audio'].float().cuda()
        outputs = model2(inputs)
        _, preds = torch.max(outputs, 1)
        mapper = list(
            map(
                lambda i: {
                    'label': sample['label'][i],
                    'class': preds[i].detach().cpu().numpy().tolist()
                }, range(len(sample['label']))))
        audio_rel.append(mapper)

    print('Angle calculated...')
    angle_rel = []
    test_data3 = ActionTestDataset(root_path)
    test_loader3 = DataLoader(test_data3, batch_size=1)
    model3 = ActionAngle().load_from_checkpoint(
        'weights/action-angle-resnet110.ckpt')
    model3.cuda()
    model3.eval()
    for sample in test_loader3:
        inputs = sample['audio'].float().cuda()
        outputs = model3(inputs)
        tmp = list(
            map(
                lambda i: {
                    'label': sample['label'][i],
                    'angle_net': outputs[i].detach().cpu().numpy().tolist()
                }, range(len(sample['label']))))
        angle_rel.append(tmp)

    print('Location calculated...')
    loc_rel = []
    model4 = ActionLoc().load_from_checkpoint(
        'weights/action-loc-resnet110.ckpt')
    model4.cuda()
    model4.eval()
    for sample in test_loader3:
        inputs = sample['audio'].float().cuda()
        outputs = model4(inputs)
        tmp = list(
            map(
                lambda i: {
                    'label': sample['label'][i],
                    'pos_net': outputs[i].detach().cpu().numpy().tolist()
                }, range(len(sample['label']))))
        loc_rel.append(tmp)
    print('Start matching...')
    angle_rel = pd.json_normalize(sum(angle_rel, []))
    audio_rel = pd.json_normalize(sum(audio_rel, []))
    image_rel = pd.json_normalize(sum(image_rel, []))
    loc_rel = pd.json_normalize(sum(loc_rel, []))
    mask_list = pd.json_normalize(mask_list)

    audio_info = pd.merge(pd.merge(audio_rel,
                                   angle_rel,
                                   on='label',
                                   how='outer'),
                          loc_rel,
                          on='label',
                          how='outer')
    video_info = pd.merge(image_rel, mask_list, on='label', how='outer')
    info_list = pd.merge(audio_info, video_info, on='class', how='outer')

    classes = info_list.groupby(['class'])
    info_classes = [classes.get_group(i).to_dict('records') for i in range(10)]
    results = [find_match(d) for d in info_classes]
    results = sorted(sum(results, []))
    results = dict(
        list(map(lambda i: (i[0], int(i[1].split('_')[-1])), results)))
    unmatched_audio = [d
                       for l in test_data2.data_dir if l['label'] not in results
                       for d in audio_info.to_dict('records') if d['label'] == l['label']]
    unmatched_video = [d
                       for l in test_data1.data_dir if int(
                           l['label'].split('_')[-1]) not in results.values()
                       for d in video_info.to_dict('records') if d['label'] == l['label']]
    unmatched_info = \
        pd.json_normalize(unmatched_audio) \
        .assign(key=1) \
        .merge(pd.json_normalize(unmatched_video)
               .assign(key=1),
               on='key',
               how='outer') \
        .drop('key', axis=1) \
        .to_dict('records')
    unmatched_results = dict(
        list(
            map(
                lambda i: (i[0], int(i[1].split('_')[-1])),
                sorted(find_match(unmatched_info)))))
    results.update(unmatched_results)
    print('----------Finish task2 test----------')

    return results


def test_task3(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task3/test/0/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': -1, ‘audio_0001’: 12, ...}
    This means audio 'audio_0000.pkl' is not matched to any video and ‘audio_0001’ is matched to 'video_0012'.
    '''
    print('----------Start task3 test----------')
    print('Pre-processing audio data!')
    pre_process(root_path)

    print('Processing mask data...')
    mask_list = test_mask_process(root_path)

    print('Image classification...')
    image_rel = []
    test_data1 = VideoTestDataset(root_path)
    test_loader1 = DataLoader(test_data1, batch_size=16)
    model1 = ImageClassifier().load_from_checkpoint(
        'weights/image-resnet44.ckpt')
    model1.cuda()
    model1.eval()
    for sample in test_loader1:
        inputs = sample['rgb'].float().cuda()
        outputs = model1(inputs)
        _, preds = torch.max(outputs, 1)
        tmp = list(
            map(
                lambda i: {
                    'label': sample['label'][i],
                    'class': preds[i].detach().cpu().numpy().tolist()
                }, range(len(sample['label']))))
        image_rel.append(tmp)

    print('Audio classification...')
    audio_rel = []
    test_data2 = AudioTestDataset(root_path)
    test_loader2 = DataLoader(test_data2, batch_size=1)
    model2 = AudioClassifier().load_from_checkpoint(
        'weights/audio-resnet20.ckpt')
    model2.cuda()
    model2.eval()
    for sample in test_loader2:
        inputs = sample['audio'].float().cuda()
        outputs = model2(inputs)
        _, preds = torch.max(outputs, 1)
        mapper = list(
            map(
                lambda i: {
                    'label': sample['label'][i],
                    'class': preds[i].detach().cpu().numpy().tolist()
                }, range(len(sample['label']))))
        audio_rel.append(mapper)

    print('Angle calculated...')
    angle_rel = []
    test_data3 = ActionTestDataset(root_path)
    test_loader3 = DataLoader(test_data3, batch_size=1)
    model3 = ActionAngle().load_from_checkpoint(
        'weights/action-angle-resnet110.ckpt')
    model3.cuda()
    model3.eval()
    for sample in test_loader3:
        inputs = sample['audio'].float().cuda()
        outputs = model3(inputs)
        tmp = list(
            map(
                lambda i: {
                    'label': sample['label'][i],
                    'angle_net': outputs[i].detach().cpu().numpy().tolist()
                }, range(len(sample['label']))))
        angle_rel.append(tmp)

    print('Location calculated...')
    loc_rel = []
    model4 = ActionLoc().load_from_checkpoint(
        'weights/action-loc-resnet110.ckpt')
    model4.cuda()
    model4.eval()
    for sample in test_loader3:
        inputs = sample['audio'].float().cuda()
        outputs = model4(inputs)
        tmp = list(
            map(
                lambda i: {
                    'label': sample['label'][i],
                    'pos_net': outputs[i].detach().cpu().numpy().tolist()
                }, range(len(sample['label']))))
        loc_rel.append(tmp)
    print('Start matching...')
    angle_rel = pd.json_normalize(sum(angle_rel, []))
    audio_rel = pd.json_normalize(sum(audio_rel, []))
    image_rel = pd.json_normalize(sum(image_rel, []))
    loc_rel = pd.json_normalize(sum(loc_rel, []))
    mask_list = pd.json_normalize(mask_list)

    audio_info = pd.merge(pd.merge(audio_rel,
                                   angle_rel,
                                   on='label',
                                   how='outer'),
                          loc_rel,
                          on='label',
                          how='outer')
    video_info = pd.merge(image_rel, mask_list, on='label', how='outer')
    info_list = pd.merge(audio_info, video_info, on='class', how='outer')

    classes = info_list.groupby(['class'])
    info_classes = [classes.get_group(i).to_dict('records') for i in range(10)]
    results = [find_match(d) for d in info_classes]
    results = sorted(sum(results, []))
    results = dict(
        list(map(lambda i: (i[0], int(i[1].split('_')[-1])), results)))
    for l in test_data2.data_dir:
        results.setdefault(l['label'], -1)
    print('----------Finish task3 test----------')
    return results


if __name__ == '__main__':
    task1 = test_task1('./dataset/task1/test')
    # json.dump(task1, open('task1_results.json', 'w', encoding='utf-8'), ensure_ascii=False)
    task2 = test_task2('./dataset/task2/test/3')
    # json.dump(task2, open('task2_results.json', 'w', encoding='utf-8'), ensure_ascii=False)
    task3 = test_task3('./dataset/task3/test/3')
    # json.dump(task3, open('task3_results.json', 'w', encoding='utf-8'), ensure_ascii=False)
