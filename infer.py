import torch
import torchaudio
from bsrnn import BSRNN
from m_dataset import load_waveform

import argparse

parser = argparse.ArgumentParser(description='Infer the BSRNN model')
parser.add_argument("--input", type=str, help="Input file")
parser.add_argument("--output", type=str, help="Output file")
args = parser.parse_args()

model = BSRNN().to("cuda")
model.load_state_dict(torch.load("model.pth"))
model.eval()

waveform, sr = torchaudio.load(args.input)
waveform = waveform.to("cuda")
if waveform.shape[0] == 1:
    waveform = torch.cat((waveform, waveform), 0)

stft_window = torch.hann_window(4096).to("cuda")
waveform = torch.stft(waveform, n_fft=4096, hop_length=512, return_complex=True, window=stft_window)
x = model.forward(waveform)
x = torch.istft(x, n_fft=4096, hop_length=512, window=stft_window)

x = x.cpu()
torchaudio.save(args.output, x, sr)
