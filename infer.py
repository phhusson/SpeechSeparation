import numpy as np
import torch
import torch.nn
import torchaudio
from bsrnn import BSRNN
from m_dataset import load_waveform

import argparse

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='Infer the BSRNN model')
parser.add_argument("--input", type=str, help="Input file")
parser.add_argument("--output", type=str, help="Output file")
args = parser.parse_args()

model = BSRNN().to("cpu")
model.load_state_dict(torch.load("model.pth"))
model.eval()
#torch.onnx.dynamo_export(model, torch.ones([2, 2049, 512]))
#torch.onnx.export(model, torch.ones([2, 2049, 512], dtype = torch.complex64), "hello.onnx", opset_version=15)

waveform, sr = torchaudio.load(args.input)
waveform = waveform.to("cpu")
if waveform.shape[0] == 1:
    waveform = torch.cat((waveform, waveform), 0)

stft_window = torch.hann_window(4096).to("cpu")
orig_peak = waveform.max().item()
x = torch.stft(waveform, n_fft=4096, hop_length=512, return_complex=True, window=stft_window)
x = model.forward(x)
x = torch.istft(x, n_fft=4096, hop_length=512, window=stft_window)

x = x.cpu()
waveform = waveform.cpu()
waveform = waveform[:,:x.shape[1]]
torchaudio.save(args.output, x, sr)

aaa, sr = torchaudio.load("/home/phh/d/d/d4/dnr_v2/tt/10020/speech.wav")
torchaudio.save("delta-" + args.output, (waveform - x), sr)

signalPower = np.sum(np.square(waveform.numpy()))
remainingPower = np.sum(np.square((waveform - x).numpy()))

print("Separation dB", 10 * np.log(signalPower / remainingPower))

dialog = x
inst = waveform[:,:dialog.shape[1]] - x
if True:
    mix = dialog
    mix = mix * orig_peak / (mix.max())
    torchaudio.save('mix_100.wav', mix, sr)
    del mix

if True:
    mix = dialog + 0.3 * inst
    mix = mix * orig_peak / (mix.max())
    torchaudio.save('mix_90.wav', mix, sr)
    del mix

if True:
    mix = dialog + 0.5 * inst
    mix = mix * orig_peak / (mix.max())
    torchaudio.save('mix_50.wav', mix, sr)
    del mix

if True:
    mix = dialog + 0.8 * inst
    mix = mix * orig_peak / (mix.max())
    torchaudio.save('mix_20.wav', mix, sr)
    del mix

if True:
    mix = inst
    mix = mix * orig_peak / (mix.max())
    torchaudio.save('mix_-100.wav', mix, sr)
    del mix
