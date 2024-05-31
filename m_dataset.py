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
    x = model.forward(waveform)
    waveform_speech = waveform_speech[:, :x.shape[1]]
    return x, waveform_speech

def train_infer(model, sample, lossfn, verbose=False):
    x, waveform_speech = infer(model, sample)

    if verbose:
        print("x", x.mean().item(), x.max().item(), x.std().item(), x.min().item())
        print("waveform_speech", waveform_speech.mean().item(), waveform_speech.max().item(),
              waveform_speech.std().item(), waveform_speech.min().item())

    loss = lossfn(x, waveform_speech)
    sdr = 10 * torch.log10(
        torch.linalg.vector_norm(waveform_speech, ord=1) / (x.shape[1] * loss + 1e-9))
    return loss, sdr
