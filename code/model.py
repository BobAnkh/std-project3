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


class AudioClassifier(nn.Module):
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
        self.audio_embed = AudioEmbed()
        self.video_embed = VideoEmbed()
        self.audio_classify = AudioClassifier()
        self.similarity = Similarity()

    def forward(self, x):
        pass
