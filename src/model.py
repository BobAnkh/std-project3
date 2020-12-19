import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from . import ConvLSTM
from . import resnet
from . import alexnet


class AudioEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet.resnet34()
        # self.alexnet = alexnet.AlexNet()

    def forward(self, x):
        output = self.resnet(x)
        # output = self.alexnet(x)
        return output


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
