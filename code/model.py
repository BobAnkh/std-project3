import torch.nn as nn


class AudioEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class VideoEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class AudioClassify(nn.Module):
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
        self.sound_embed = AudioEmbed()
        self.video_embed = VideoEmbed()
        self.sound_classify = AudioClassify()
        self.similarity = Similarity()

    def forward(self, x):
        pass
