import numpy as np
import torch
import torch.nn
import torchaudio
from bsrnn import BSRNN
from m_dataset import load_waveform

import argparse
import sys

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='Infer the BSRNN model')
parser.add_argument("--input", type=str, help="Input file")
parser.add_argument("--output", type=str, help="Output file")
args = parser.parse_args()

class ToTrace(torch.nn.Module):
    def __init__(self, model):
        super(ToTrace, self).__init__()
        self.model = model

    def forward(self, x, state):
        return self.model.forward_recurrent(x, state)


model = BSRNN().to("cpu")
model.load_state_dict(torch.load("model.pth"))
model.eval()

tmodel = ToTrace(model)
# Compile the model to torchscript for faster execution using tracing
example_input = (torch.ones([2, 2049], dtype=torch.complex64), torch.zeros((2, 2, 2, 36, 128)))
model = torch.jit.trace(tmodel, example_input)

#torch.onnx.dynamo_export(model, torch.ones([2, 2049, 512]))
#torch.onnx.export(model, example_input, "hello.onnx", opset_version=15)

streamer = torchaudio.io.StreamReader(args.input)
streamer.add_basic_audio_stream(frames_per_chunk=512, sample_rate=44100)

writer = torchaudio.io.StreamWriter(args.output)
writer.add_audio_stream(sample_rate=44100, num_channels=2)
writer.open()

stft_window = torch.hann_window(4096).to("cpu")
current_waveform = torch.zeros( (2, 4096) )
# TODO: How to retrieve this shape automatically?
# 2 Timewise LSTM x (c_x,h_x) x LSTM state
state = torch.zeros((2, 2, 2, 36, 128))
# With 512/4096 hop there are 4 blocks left of center.
# The next 512 new samples are the 512 after center of current
# the 512 before center of current-1
# the 512 off 512 of current-2, etc
previous_speech = [torch.zeros( (2, 4096))] * 8
for chunks in streamer.stream():

    # chunks[0] is [512; 2]
    waveform = chunks[0].permute( (1, 0) )
    # Mono to stereo
    if waveform.shape[0] == 1:
        waveform = torch.cat((waveform, waveform), 0)

    waveform = torch.cat( (current_waveform[:,512:], waveform), 1)
    current_waveform = waveform

    x = torch.fft.rfft(waveform * stft_window)
    x = x.reshape((2, -1))
    #x, state = model.forward_recurrent(x, state)
    x, state = model(x, state)
    waveform = torch.fft.irfft(x)
    
    previous_speech = [
            waveform,
            previous_speech[0],
            previous_speech[1],
            previous_speech[2],
            previous_speech[3],
            previous_speech[4],
            previous_speech[5],
            previous_speech[6],
    ]
    sum_of_window = torch.zeros( 512 )
    new_samples = torch.zeros( (2, 512) )
    current_range = (0, 512)
    for wf in previous_speech:
        win = stft_window[current_range[0]:current_range[1]]
        sum_of_window = sum_of_window + win
        new_samples = new_samples + wf[:, current_range[0]:current_range[1]]
        current_range = (current_range[0] + 512, current_range[1] + 512)

    new_samples = new_samples / sum_of_window

    writer.write_audio_chunk(0, new_samples.permute( (1,0)) )

sys.exit(0)

waveform, sr = torchaudio.load(args.input)
waveform = waveform.to("cpu")
if waveform.shape[0] == 1:
    waveform = torch.cat((waveform, waveform), 0)

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

