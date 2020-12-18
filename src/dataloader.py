from PIL import Image
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


class trainDataset(torch.utils.data.Dataset):
    def __init__(self, path: str = "./dataset/train", audio: bool = True, video: bool = True) -> None:
        super().__init__()
        self.path = path
        self.audio = audio
        self.video = video
        self.labels = os.listdir(self.path)
        self.data_dir = [{"class_name": c, "class": label[c], "label": i, "dir": os.path.join(self.path, c, i)}
                         for c in self.labels if not re.match(r".+\.rar", c)
                         for i in os.listdir(os.path.join(self.path, c))]

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        data_dir = self.data_dir[idx]
        data = {"class": data_dir["class"], "label": data_dir["label"]}
        if self.audio:
            audio = np.load(os.path.join(data_dir["dir"], "audio_data.npy"))
            data["audio"] = audio

        if self.video:
            # mask_img_dir = [os.path.join(data_dir["dir"], "mask", i) for i in sorted(os.listdir(
            #     os.path.join(data_dir["dir"], "mask")))]
            # mask = torch.as_tensor([np.asarray(Image.open(path))
            #                         for path in mask_img_dir])
            # data["mask"] = mask
            rgb_img_dir = [os.path.join(data_dir["dir"], "rgb", i) for i in sorted(os.listdir(
                os.path.join(data_dir["dir"], "rgb")))]
            rgb = torch.as_tensor([np.asarray(Image.open(path))
                                   for path in rgb_img_dir])
            rgb = rgb.permute(0, 3, 1, 2)
            data["rgb"] = rgb

        return data
