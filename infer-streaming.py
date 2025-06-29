import numpy as np
import torch
import time
import torch.nn
import torchaudio
from bsrnn import BSRNN
from m_dataset import load_waveform
from torch.utils.mobile_optimizer import optimize_for_mobile
import argparse
import sys

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='Infer the BSRNN model')
parser.add_argument("--input", type=str, help="Input file")
parser.add_argument("--output", type=str, help="Output file")
parser.add_argument("--name", type=str, help="Model name")
args = parser.parse_args()

class ToTrace(torch.nn.Module):
    def __init__(self, model):
        super(ToTrace, self).__init__()
        self.model = model

    def forward(self, x, state):
        (a, b) = self.model.forward_recurrent(x, state)
        return (a,b,torch.tensor([]))

class ToTrace1(torch.nn.Module):
    def __init__(self, model):
        super(ToTrace1, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_recurrent_first_half(x)

class ToTrace2(torch.nn.Module):
    def __init__(self, model):
        super(ToTrace2, self).__init__()
        self.model = model

    def forward(self, x, bands, state):
        return self.model.forward_recurrent_second_half(x, bands, state)

model = BSRNN().to("cpu")
model.load_state_dict(torch.load("model-always.pth"))
model.eval()

torch.save(model, "model-full.pth")
tmodel = ToTrace(model)
tmodel1 = ToTrace1(model)
tmodel2 = ToTrace2(model)
# Compile the model to torchscript for faster execution using tracing
fft_window = 2 * 1024
example_input = (torch.ones([2, (fft_window // 2 + 1) * 2]), torch.zeros((4, 2, 24, 64)))
model = tmodel
#model = torch.jit.trace(tmodel, example_input)
#
#input1 = torch.ones([2, 1025 * 2])
#model1 = torch.jit.trace(tmodel1, input1)
#model1 = optimize_for_mobile(model1)
#torch.jit.save(model1, "model-script-1.pth")
#
#input2 = (torch.ones([2, 1025 * 2]), torch.ones([2, 10, 64]), torch.zeros((4, 2, 36, 64)))
#model2 = torch.jit.trace(tmodel2, input2)
#model2 = optimize_for_mobile(model2)
#torch.jit.save(model2, "model-script-2.pth")
#
#model = optimize_for_mobile(model)
#torch.jit.save(model, "model-script.pth")
#
##onnx_program = torch.onnx.dynamo_export(tmodel, example_input[0], example_input[1])
##onnx_program.save("model.onnx")
torch.onnx.export(model, example_input, "hello.onnx", opset_version=17, input_names = ["x.0", "state.0"], output_names = ["y.0", "new_state.0", f"model={args.name}"])
print("Onnx model dumped")

streamer = torchaudio.io.StreamReader(args.input)
streamer.add_basic_audio_stream(frames_per_chunk=1024, sample_rate=44100)

writer = torchaudio.io.StreamWriter(args.output)
writer.add_audio_stream(sample_rate=44100, num_channels=2)
writer.open()

stft_window = torch.hann_window(fft_window).to("cpu")
current_waveform = torch.zeros( (2, fft_window) )
# TODO: How to retrieve this shape automatically?
# 2 Timewise LSTM x (c_x,h_x) x LSTM state
state = torch.zeros((4, 2, 24, 64))
# With 512/4096 hop there are 4 blocks left of center.
# The next 512 new samples are the 512 after center of current
# the 512 before center of current-1
# the 512 off 512 of current-2, etc
previous_speech = [torch.zeros( (2, fft_window))] * 8

t = time.time()
for chunks in streamer.stream():

    # chunks[0] is [512; 2]
    if chunks[0].shape[0] != 1024:
        break
    waveform = chunks[0].permute( (1, 0) )
    # Mono to stereo
    if waveform.shape[0] == 1:
        waveform = torch.cat((waveform, waveform), 0)

    waveform = torch.cat( (current_waveform[:,1024:], waveform), 1)
    current_waveform = waveform

    x = torch.fft.rfft(waveform * stft_window)
    x = torch.stack((x.real, x.imag), dim=2)
    x = x.reshape((2, -1))

    x, state, bip = model(x, state)
    x = x.reshape((2,  -1, 2))
    x = torch.complex(x[:, :, 0], x[:, :, 1])

    waveform = torch.fft.irfft(x)
    
    previous_speech = [
            waveform,
            previous_speech[0],
    ]
    sum_of_window = torch.zeros( 1024 )
    new_samples = torch.zeros( (2, 1024) )
    current_range = (0, 1024)
    for wf in previous_speech:
        win = stft_window[current_range[0]:current_range[1]]
        sum_of_window = sum_of_window + win
        new_samples = new_samples + wf[:, current_range[0]:current_range[1]]
        current_range = (current_range[0] + 1024, current_range[1] + 1024)

    new_samples = new_samples / sum_of_window

    writer.write_audio_chunk(0, new_samples.permute( (1,0)) )

print(f"Elapsed {time.time() - t}")
