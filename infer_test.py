import torch
from bsrnn import BSRNN

from m_dataset import samples, MyDataSet, train_infer
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='Test the BSRNN model')
parser.add_argument('--datapath', type=str, default='/home/phh/d/d/d4/dnr_v2', help='Path to the dataset')
args = parser.parse_args()

model = BSRNN().to("cuda")
model.load_state_dict(torch.load("model.pth"))
model.eval()

test = samples(args.datapath, 'tt')
test = MyDataSet(test)
test_loader = DataLoader(test, batch_size=1, shuffle=True)

if True:
    # Explore the first sample
    sample = test[0]
    waveform, waveform_speech = sample
    waveform = waveform.squeeze(0)
    waveform_speech = waveform_speech.squeeze(0)
    x = model.forward(waveform)
    waveform_speech = waveform_speech[:, :x.shape[1]]

    print("x", x.mean().item(), x.max().item(), x.std().item(), x.min().item())
    print("waveform_speech", waveform_speech.mean().item(), waveform_speech.max().item(),
          waveform_speech.std().item(), waveform_speech.min().item())

loss = 0.0
sdr = 0.0
for sample in test:
    l, s = train_infer(model, sample, torch.nn.L1Loss())
    loss += l.item()
    sdr += s.item()

print("Loss", loss / len(test))
print("SDR", sdr / len(test))
