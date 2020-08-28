from doggo_ears_definitions import *

import torch.nn as nn

class classifier(nn.Module):

    def __init__(self, input_size, n_cat, hiddensize=50):
        super(classifier, self).__init__()
        self.unit = nn.LSTM(input_size, hiddensize, 
                num_layers=1, batch_first = True, dropout = 0.05)
        self.fc = nn.Linear(hiddensize,n_cat)

    def forward(self, x):
        outputs, (h, c) = self.unit(x)
        outputs = self.fc(h[-1,:,:])
        return outputs

    def pred(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.single))
        return nn.functional.softmax(self.forward(x.unsqueeze(0)).squeeze()).tolist()