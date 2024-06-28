import numpy as np
import math
import torch
import time
import torch.nn
import torchaudio
from bsrnn import BSRNN
from m_dataset import load_waveform
from torch.utils.mobile_optimizer import optimize_for_mobile
import argparse
import sys
import os

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='Infer the BSRNN model')
parser.add_argument("--input", type=str, help="Input file")
parser.add_argument("--output", type=str, help="Output dir")
parser.add_argument("--channel", type=str, help="Channel to use")
args = parser.parse_args()

channel = int(args.channel)

class ToTrace(torch.nn.Module):
    def __init__(self, model):
        super(ToTrace, self).__init__()
        self.model = model

    def forward(self, x, state):
        return self.model.forward_recurrent(x, state)

class Separator():
    def __init__(self):
        model = BSRNN().to("cpu")
        model.load_state_dict(torch.load("model.pth"))
        model.eval()
        self.model = ToTrace(model)
        self.state = torch.zeros((4, 2, 10, 64))
        self.current_waveform = torch.zeros( (1, 4096) )
        self.stft_window = torch.hann_window(4096).to("cpu")
        self.previous_speech = [torch.zeros( (1, 4096))] * 4

    def write(self, x):
        # x is [512]
        # chunks[0] is [512; 2]
        if x.shape[0] != 1024:
            return None
        waveform = x.unsqueeze(0)

        waveform = torch.cat( (self.current_waveform[:,1024:], waveform), 1)
        self.current_waveform = waveform

        x = torch.fft.rfft(waveform * self.stft_window)
        x = torch.stack((x.real, x.imag), dim=2)
        x = x.reshape((1, -1))

        x, self.state = self.model(x, self.state)
        x = x.reshape((1, -1, 2))
        x = torch.complex(x[:, :, 0], x[:, :, 1])

        waveform = torch.fft.irfft(x)
        
        self.previous_speech = [
                waveform,
                self.previous_speech[0],
                self.previous_speech[1],
                self.previous_speech[2],
        ]
        sum_of_window = torch.zeros( 1024 )
        new_samples = torch.zeros( (1, 1024) )
        current_range = (0, 1024)
        for wf in self.previous_speech:
            win = self.stft_window[current_range[0]:current_range[1]]
            sum_of_window = sum_of_window + win
            new_samples = new_samples + wf[:, current_range[0]:current_range[1]]
            current_range = (current_range[0] + 1024, current_range[1] + 1024)

        new_samples = new_samples / sum_of_window
        return new_samples.squeeze(0)


streamer = torchaudio.io.StreamReader(args.input)
streamer.add_basic_audio_stream(frames_per_chunk=1024, sample_rate=44100)

separator = Separator()
writer = None

t = time.time()
nDialog = 0
nBackground = 0
i = 0
for chunks in streamer.stream():
    i += chunks[0].shape[0]
    # chunks[0] is [512; 6]
    if chunks[0].shape[0] != 1024:
        break
    waveform = chunks[0][:,channel]

    dialog = separator.write(waveform)
    waveform = waveform[:dialog.shape[0]]
    background = waveform - dialog

    dialogToBackground = (torch.square(dialog) / torch.square(background)).mean()
    db = 10 * math.log(dialogToBackground)
    meanLevel = torch.square(waveform).mean()
    if meanLevel < 1e-10:
        level = -200
    else:
        level = 10 * math.log(meanLevel)

    if db > 10:
        # This is a dialog section
        if nBackground > 0:
            nBackground = 0
            if writer:
                writer.close()
                writer = None

        nDialog += 1
        if nDialog >= 10:
            if not writer:
                writer = torchaudio.io.StreamWriter(args.output + f'/dialog-{i}.wav')
                writer.add_audio_stream(sample_rate=44100, num_channels=1)
                writer.open()
            writer.write_audio_chunk(0, waveform.unsqueeze(1))
    elif (db > 3 or level < -100) and nDialog >= 10:
        if writer:
            writer.write_audio_chunk(0, waveform.unsqueeze(1))
    elif db  < -30:
        # this is a background section
        if nDialog > 0:
            nDialog = 0
            if writer:
                writer.close()
                writer = None

        nBackground += 1
        if nBackground >= 10:
            if not writer:
                writer = torchaudio.io.StreamWriter(args.output + f'/background-{i}.wav')
                writer.add_audio_stream(sample_rate=44100, num_channels=1)
                writer.open()
            writer.write_audio_chunk(0, waveform.unsqueeze(1))
    else:
        nDialog = 0
        nBackground = 0
        if writer:
            writer.close()
            writer = None

print(f"Elapsed {time.time() - t}")
