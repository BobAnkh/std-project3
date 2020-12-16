import pickle as pkl
import numpy as np
import librosa

SR_DOWN = 11000


def audio_process(path):
    """
    按照论文中的方式对声音数据进行预处理
    - 没有做log transform
    """
    data = pkl.load(open(path, 'rb'))
    audio = data['audio'].T
    sr = data['audio_samplerate']

    stft_data = []
    for a in audio:
        a_rs = librosa.resample(a, sr, SR_DOWN)
        a_stft = librosa.stft(a_rs, n_fft=510, hop_length=128)
        stft_data.append(a_stft)

    return stft_data


def main():
    import matplotlib.pyplot as plt
    from librosa.display import specshow

    stft = audio_process('data/train/toy_elephant/0/audio_data.pkl')

    # display stft result
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    for i in range(4):
        D = librosa.amplitude_to_db(np.abs(stft[i]), ref=np.max)
        img = specshow(D, y_axis='linear', x_axis='time', sr=SR_DOWN, ax=ax[i])
        ax[i].label_outer()
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()


if __name__ == '__main__':
    main()
