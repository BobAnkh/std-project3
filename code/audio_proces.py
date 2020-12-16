import pickle as pkl

import librosa
import numpy as np

SR_DOWN = 11000


def audio_process(path):
    """
    按照论文中的方式对声音数据进行预处理
    """
    data = pkl.load(open(path, 'rb'))
    audio = data['audio']
    sr = data['audio_samplerate']

    stft_data = []
    for a in audio:
        a_rs = librosa.resample(a, sr, SR_DOWN)
        a_stft = librosa.stft(a_rs, n_fft=510, hop_length=128)
        stft_data.append(a_stft)


def main():
    audio_process('data/audio_data.pkl')


if __name__ == '__main__':
    main()
