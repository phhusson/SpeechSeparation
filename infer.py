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
x = model.forward(waveform)
x = x.cpu()
torchaudio.save(args.output, x, sr)
