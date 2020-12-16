import torch.nn as nn


class SoundEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class VideoEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class SoundClassify(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Similarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.sound_embed = SoundEmbed()
        self.video_embed = VideoEmbed()
        self.sound_classify = SoundClassify()
        self.similarity = Similarity()

    def forward(self, x):
        pass
