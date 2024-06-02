import os
import torchaudio
import torch

def samples(sample_bases, folder):
    s = [sample_bases + '/' + folder + '/' + x for x in os.listdir(sample_bases + '/' + folder)]
    # Filter out files
    s = [x for x in s if os.path.isdir(x)]
    s = [(x + '/mix.wav', x + '/speech.wav') for x in s]
    return s


def load_waveform(path):
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] == 1:
        waveform = torch.cat((waveform, waveform), 0)
    return waveform.to("cuda")


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, s):
        self.samples = s

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return load_waveform(self.samples[idx][0]), load_waveform(self.samples[idx][1])

def infer(model, sample):
    waveform, waveform_speech = sample
    waveform = waveform.squeeze(0)
    waveform_speech = waveform_speech.squeeze(0)

    stft_window = torch.hann_window(4096).to("cuda")
    waveform = torch.stft(waveform, n_fft=4096, hop_length=512, return_complex=True, window=stft_window)
    x = model.forward(waveform)
    x_time = torch.istft(x, n_fft=4096, hop_length=512, window=stft_window)
    waveform_speech_freq = torch.stft(waveform_speech, n_fft=4096, hop_length=512, return_complex=True, window=stft_window)

    waveform_speech = waveform_speech[:, :x_time.shape[1]]
    return x, x_time, waveform_speech_freq, waveform_speech

def train_infer(model, sample, lossfn, verbose=False):
    x_freq, x_time, waveform_speech_freq, waveform_speech_time = infer(model, sample)

    loss = (lossfn(x_time, waveform_speech_time) +
            lossfn(x_freq.real, waveform_speech_freq.real) +
            lossfn(x_freq.imag, waveform_speech_freq.imag))
    n2Source = torch.sum(torch.square(waveform_speech_time), dim=1)
    n2Delta = torch.sum(torch.square(x_time - waveform_speech_time), dim=1)
    sdr = 10 * torch.log10(n2Source / n2Delta)
    sdr = sdr.mean()
    return loss, sdr
