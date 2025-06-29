import os
import random
import torchaudio
import torchmetrics
import torch
import soundfile

def samples(sample_bases, folder, with_s2s = False, with_rnnoise = False):
    s = [sample_bases + '/' + folder + '/' + x for x in os.listdir(sample_bases + '/' + folder)]
    # Filter out files
    sdirs = [x for x in s if os.path.isdir(x)]
    #s = [(x + '/mix.wav', x + '/speech_post_rnnoise.wav') for x in sdirs]
    speech = "/speech.wav"
    if with_rnnoise:
        speech = "/speech_post_rnnoise.wav"
    s = [(x + '/mix.wav', x + speech) for x in sdirs]
    if with_s2s:
        s += [(x + speech,  x + speech) for x in sdirs]
    return s

def mix_samples(base):
    if isinstance(base, list):
        retDialogs = []
        retBackgrounds = []
        for x in base:
            a, b = mix_samples(x)
            retDialogs += a
            retBackgrounds += b
        return (retDialogs, retBackgrounds)
    dialogs = []
    backgrounds = []
    for x in os.listdir(base):
        p = base + '/' + x
        if os.path.isdir(p):
            ret = mix_samples(p)
            dialogs += ret[0]
            backgrounds += ret[1]
        elif os.path.isfile(p):
            #if x.startswith('dialog-'):
            #    dialogs += [p]
            if 'lownoise' in p and p.endswith(".wav"):
                dialogs += [p]
            elif x.startswith('speech.wav'):
                dialogs += [p]
            elif x.startswith('speech_post_rnnoise.wav'):
                dialogs += [p]
            #if x.startswith('background-'):
            #    backgrounds += [p]
            elif x.startswith('sfx.wav'):
                backgrounds += [p]
            elif x.startswith('music.wav'):
                backgrounds += [p]
            elif x.startswith('vocalnoise') and endswith('.wav'):
                backgrounds += [p]
    return (dialogs, backgrounds)

def load_waveform(path):
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] == 1:
        waveform = torch.cat((waveform, waveform), 0)
    return waveform.to("cuda")

def load_waveform_cpu(path):
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] == 1:
        waveform = torch.cat((waveform, waveform), 0)
    return waveform

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, s, train = False):
        self.samples = s
        self.train = train
        self.resampler = torchaudio.transforms.Resample(16000, 44100).to('cuda')
        if train:
            rir_base = "/nvme1/ML/DnR/RIRS_NOISES/real_rirs_isotropic_noises"
            self.rirs = [rir_base + '/' + x for x in os.listdir(rir_base) if "imp" in x]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        srcf = self.samples[idx][0]
        random.seed(idx)
        # 10% chance of picking poorly encoded input instead of wav at train time
        #if self.train:
        #    res = random.choices([True, False], weights=[0.1, 0.9])[0]
        #    if res:
        #        srcf = srcf.replace("mix.wav", "mix_opus_8k.wav")

        a = load_waveform(srcf)

        # Randomly apply a RIR
        if self.train:
            # 10% chance of applying a rir
            res = random.choices([True, False], weights=[0.1, 0.9])[0]
            if res:
                rir = random.choices(self.rirs)[0]
                rir, sr = soundfile.read(rir)
                rir = torch.tensor(rir).to('cuda')
                # Pick two random channels of the RIR
                rir_channel1 = random.randint(0, rir.shape[1]-1)
                rir_channel2 = random.randint(0, rir.shape[1]-1)
                rir1 = rir[:,rir_channel1]
                rir2 = rir[:,rir_channel2]
                rir1 = self.resampler(rir1.float())
                rir2 = self.resampler(rir2.float())

                a1 = a[0]
                a2 = a[1]
                a1 = torch.nn.functional.conv1d(a1.reshape(1, 1, -1), rir1.reshape(1, 1, -1), padding = 'same').reshape(1, -1)
                a2 = torch.nn.functional.conv1d(a1.reshape(1, 1, -1), rir2.reshape(1, 1, -1), padding = 'same').reshape(1, -1)
                a1 = a1[:, :a.shape[1]]
                a2 = a2[:, :a.shape[1]]
                a = torch.cat((a1,a2), 0)

        voice_filename = self.samples[idx][1]
        b = load_waveform(voice_filename)
        if 'rnnoise' in voice_filename:
            b = torch.roll(b, -480, dims = 1)
        l = min(a.shape[1], b.shape[1])
        return a[:, :l], b[:, :l]

def pad(left, right, x):
    padding = torch.zeros( (x.shape[0], 1))
    a = padding.repeat((1,left)).to(x.device)
    b = padding.repeat((1,right)).to(x.device)
    return torch.cat((a,x,b), dim=1)

def random_resize(x, length):
    if x.shape[1] < length:
        missing = length - x.shape[1]
        pad_left = random.randint(0, missing)
        pad_right = missing- pad_left
        x = pad(pad_left, pad_right, x)
    else:
        to_delete = x.shape[1] - length
        to_remove_left = random.randint(0, to_delete - 1)
        x = x[:,to_remove_left:(to_remove_left+length)]
    return x

class MixDataSet(torch.utils.data.Dataset):
    def __init__(self, s):
        self.dialogs = s[0]
        self.backgrounds = s[1]

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        random.seed(idx)
        background = random.choice(self.backgrounds)
        background2 = random.choice(self.backgrounds)
        dialog = random.choice(self.dialogs) #self.dialogs[idx]
        dialog2 = random.choice(self.dialogs)
        dialog3 = random.choice(self.dialogs)
        dialog4 = random.choice(self.dialogs)

        n_backgrounds = 4
        sr = 44100
        length = 50 * sr

        backgrounds = [load_waveform_cpu(random.choice(self.backgrounds)) for x in range(n_backgrounds)]
        backgrounds = [random_resize(x, length) for x in backgrounds]
        backgrounds = [x * random.uniform(0.1, 1.0) for x in backgrounds]
        dialog = load_waveform_cpu(dialog)

        ## Generated length is min(background, dialog) + 1s to have a bit of padding left and right
        #if background.shape[1] < length:
        #    length = background.shape[1] + sr
        #if dialog.shape[1] < length:
        #    length = dialog.shape[1] + sr

        scale_dialog = random.uniform(0.1, 1.0)
        #dialog *= scale_dialog

        dialog = random_resize(dialog, length)

        waveform = dialog + sum(backgrounds)

        return waveform.to('cuda'), dialog.to('cuda')

def infer(model, sample):
    waveform, waveform_speech = sample
    waveform = waveform.squeeze(0)
    waveform_speech = waveform_speech.squeeze(0)

    stft_window = torch.hann_window(2048).to("cuda")
    waveform = torch.stft(waveform, n_fft=2048, hop_length=1024, return_complex=True, window=stft_window)
    waveform = torch.stack((waveform.real, waveform.imag), dim=2)
    waveform = waveform.reshape( (waveform.shape[0], -1, waveform.shape[3]) )
    x = model.forward(waveform)
    x = x.reshape((2, -1, 2, x.shape[2]))
    x = torch.complex(x[:, :, 0, :], x[:, :, 1, :])

    x_time = torch.istft(x, n_fft=2048, hop_length=1024, window=stft_window)
    waveform_speech_freq = torch.stft(waveform_speech, n_fft=2048, hop_length=1024, return_complex=True, window=stft_window)

    waveform_speech = waveform_speech[:, :x_time.shape[1]]
    return x, x_time, waveform_speech_freq, waveform_speech

SISDR = torchmetrics.audio.ScaleInvariantSignalDistortionRatio().to('cuda')
def train_infer(model, discriminator, sample, lossfn, verbose=False):
    x_freq, x_time, waveform_speech_freq, waveform_speech_time = infer(model, sample)

    x_freq2 = torch.cat((x_freq.real, x_freq.imag), dim=1)
    if discriminator:
        with torch.no_grad():
            discriminator_score = discriminator(x_freq2)
    else:
        discriminator_score = 0
    loss = (lossfn(x_time, waveform_speech_time) +
            lossfn(x_freq.real, waveform_speech_freq.real) +
            lossfn(x_freq.imag, waveform_speech_freq.imag)) + discriminator_score
    n2Source = torch.sum(torch.square(waveform_speech_time), dim=1) + 1e-9
    n2Delta = torch.sum(torch.square( ( x_time - waveform_speech_time)), dim=1) + 1e-9
    sdr = 10 * torch.log10(n2Source / n2Delta)
    sdr = sdr.mean()

    n2Source = torch.sum(torch.square(sample[1]), dim=1) + 1e-9
    n2Delta = torch.sum(torch.square( sample[1] - sample[0]), dim=1) + 1e-9
    sdr2 = 10 * torch.log10(n2Source / n2Delta)
    sdr2 = sdr2.mean()

    sdr3 = SISDR(x_time, waveform_speech_time)

    return loss, sdr, sdr2, sdr3
