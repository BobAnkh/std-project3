from PIL import Image
from torchvision import transforms
import glob
import json
import numpy as np
import os
import re
import torch

label = {
    '061_foam_brick': 0,
    'green_basketball': 1,
    'salt_cylinder': 2,
    'shiny_toy_gun': 3,
    'stanley_screwdriver': 4,
    'strawberry': 5,
    'toothpaste_box': 6,
    'toy_elephant': 7,
    'whiteboard_spray': 8,
    'yellow_block': 9,
}

new_label = {
    '0': '061_foam_brick',
    '1': 'green_basketball',
    '2': 'salt_cylinder',
    '3': 'shiny_toy_gun',
    '4': 'stanley_screwdriver',
    '5': 'strawberry',
    '6': 'toothpaste_box',
    '7': 'toy_elephant',
    '8': 'whiteboard_spray',
    '9': 'yellow_block',
}


class AudioTrainDataset(torch.utils.data.Dataset):
    def __init__(self, path: str = "./dataset/train") -> None:
        super().__init__()
        self.path = path
        self.labels = os.listdir(self.path)
        # self.data_dir = [{"class_name": , "class": label[c], "label": i, "dir": os.path.join(self.path, c, i)}
        #                  for i in glob.glob(os.path.join(path, "**/*.pkl"), recursive=True)]
        self.data_dir = list(
            map(
                lambda tup: {
                    "class_name": tup[0][-3],
                    "class": label[tup[0][-3]],
                    "label": tup[0][-2],
                    "dir": tup[1]
                }, [(re.split(r"[/\\]", i), i)
                    for i in glob.glob(os.path.join(path, "**/*.npy"),
                                       recursive=True)]))

        self.transform = transforms.Compose(
            [transforms.ToTensor()])

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        data_dir = self.data_dir[idx]
        data = {"class": data_dir["class"], "label": data_dir["label"]}
        audio = np.load(data_dir["dir"]).transpose(1, 2, 0)
        data["audio"] = self.transform(audio)
        # audio = np.load(data_dir["dir"])
        # data["audio"] = audio

        return data


class AudioTestDataset(torch.utils.data.Dataset):
    def __init__(self, path: str = "./dataset/task1/test") -> None:
        super().__init__()
        self.path = path
        self.labels = os.listdir(self.path)
        self.data_dir = list(
            map(
                lambda tup: {
                    "label": tup[0][-1].replace('npy', 'pkl'),
                    "dir": tup[1]
                }, [(re.split(r"[/\\]", i), i)
                    for i in sorted(glob.glob(os.path.join(path, "**/*.npy"), recursive=True))]))

        self.transform = transforms.Compose(
            [transforms.ToTensor()])

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        data_dir = self.data_dir[idx]
        data = {"label": data_dir["label"]}
        audio = np.load(data_dir["dir"]).transpose(1, 2, 0)
        data["audio"] = self.transform(audio)
        return data


class VideoTrainDataset(torch.utils.data.Dataset):
    def __init__(self, path: str = "./dataset/train") -> None:
        super().__init__()
        self.path = path
        self.labels = os.listdir(self.path)
        self.data_dir = [{
            "class_name": c,
            "class": label[c],
            "label": i,
            "dir": os.path.join(self.path, c, i)
        } for c in self.labels if not re.match(r".+\.rar", c)
            for i in os.listdir(os.path.join(self.path, c))]
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize([48, 64])])

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        data_dir = self.data_dir[idx]
        data = {"class": data_dir["class"], "label": data_dir["label"]}
        # mask_img_dir = [os.path.join(data_dir["dir"], "mask", i) for i in sorted(os.listdir(
        #     os.path.join(data_dir["dir"], "mask")))]
        # mask = torch.as_tensor([np.asarray(Image.open(path))
        #                         for path in mask_img_dir])
        # data["mask"] = mask
        rgb_img_dir = [
            os.path.join(data_dir["dir"], "rgb", i)
            for i in sorted(os.listdir(os.path.join(data_dir["dir"], "rgb")))
        ]

        rgb = self.transform(Image.open(rgb_img_dir[0]))  # only load first one
        data["rgb"] = rgb

        return data


class ActionDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, basePath: str) -> None:
        """
        path: center and angle, json file
        basePath: train audio data dir path
        """
        super().__init__()
        self.path = path
        data = json.load(open(self.path))
        self.data = [{
            "class": key,
            "label": la,
            "end": np.array(li[1]),
            "angle": li[-1],
            "dir": os.path.join(basePath, new_label[key], la, "audio_data.npy")
        } for key, value in data.items() for la, li in value.items()]
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize([128, 172])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dir = self.data[idx]

        data = {"class": data_dir["class"],
                "label": data_dir["label"],
                "end": data_dir["end"],
                "angle": data_dir["angle"]}
        audio = np.load(data_dir["dir"]).transpose(1, 2, 0)
        data["audio"] = self.transform(audio)
        return data
