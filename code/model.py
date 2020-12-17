import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from . import ConvLSTM


class AudioEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class VideoEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_lstm = ConvLSTM.ConvLSTM(3, 16, (5, 5), 1, True)
        # self.conv = Conv2d(16, 32, 3)

    def forward(self, x):
        last_output, last_states = self.conv_lstm(x)

        return last_states[0][0]


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
