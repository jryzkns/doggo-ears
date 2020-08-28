from doggo_ears_definitions import *

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import librosa

class vLenDataset(Dataset):

    def __init__(self, data, labels):
        super().__init__()
        self.data = pad_sequence(data).transpose(0,1)
        self.labels = labels
        self.lengths = torch.stack([torch.tensor(data[i].shape[0])
                                        for i in range(len(data))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return  self.data[index,:,:], \
                self.labels[index], \
                self.lengths[index]

def generate_aud_features(samples):
    zero_crossings = np.mean(librosa.feature.zero_crossing_rate(samples, len(samples)))
    RMS = librosa.feature.rms(samples, len(samples)).sum()
    return zero_crossings, RMS

def corrects(preds, target):
    preds = preds.max(dim=1)[1]
    eq_vec = torch.eq(preds, target)
    return eq_vec.sum().item()

def argmax(l):
    argmax_ind, max_val = -1, -1
    for i, v in enumerate(l):
        if v > max_val:
            argmax_ind, max_val = i, v
    return argmax_ind