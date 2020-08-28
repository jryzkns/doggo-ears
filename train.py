from doggo_ears_definitions import *

from model import classifier
from utils import vLenDataset, corrects

import pickle
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("dataset.pkl", "rb") as inf_:
    dataset = pickle.load(inf_)

Y, X = zip(*dataset)
X = [torch.from_numpy(dp) for dp in X]
Y = torch.stack([label_id[label] for label in Y])

from sklearn.model_selection import train_test_split
X_tr, X_te, Y_tr, Y_te = train_test_split(X,    Y,      test_size = 0.1)
X_tr, X_va, Y_tr, Y_va = train_test_split(X_tr, Y_tr,   test_size = 0.1)

tr_loader   = DataLoader(vLenDataset(X_tr, Y_tr), **loader_params)
va_loader   = DataLoader(vLenDataset(X_va, Y_va), **loader_params)
te_loader   = DataLoader(vLenDataset(X_te, Y_te), **loader_params)

model       = classifier(N_FEATURES, N_CATEGORIES).to(device)
optimizer   = torch.optim.Adam(model.parameters(), lr=0.001)
criterion   = nn.CrossEntropyLoss() 

def train(EPOCH):
    model.train()
    tr_corrects = 0
    for batch_id, (feats, targets, lens) in enumerate(tr_loader):

        feats = pack_padded_sequence(feats, lens, 
                    batch_first=True, enforce_sorted=False)

        feats, targets = feats.to(device), targets.to(device)

        outputs = model.forward(feats)

        model.zero_grad()
        loss = criterion(outputs, targets); writer.add_scalar("Loss/train", loss, EPOCH)
        loss.backward()
        optimizer.step()

        tr_corrects += corrects(outputs, targets)

        if batch_id % 4 == 0:
            print(" ".join([f"Epoch: <{EPOCH}>", 
                        f"[ {100 * (batch_id + 1)/len(tr_loader):.3f}% ]",
                                f"Loss: {loss.data:.6f}", "\r"]), end="")
    print(f"Epoch: <{EPOCH}> [ 100.000% ] Loss: {loss.data:.6f}\r", end="")

    writer.add_scalar("Accuracy/train", 100 * tr_corrects/len(tr_loader.dataset), EPOCH)
    return 100 * tr_corrects/len(tr_loader.dataset)

def validate(EPOCH):
    model.eval()
    va_correct = 0
    for batch_ind, (feats, targets, lens) in enumerate(va_loader):
        
        feats = pack_padded_sequence(feats, lens, 
                    batch_first=True, enforce_sorted=False)

        feats, targets = feats.to(device), targets.to(device)

        outputs = model.forward(feats)
        va_correct += corrects(outputs, targets)
        loss = criterion(outputs, targets); writer.add_scalar("Loss/validation", loss, EPOCH)

    print(f"Val Accuracy: {100 * va_correct/(len(va_loader.dataset)):.4f}%")
    writer.add_scalar("Accuracy/validation", 100 * va_correct/(len(va_loader.dataset)), EPOCH)

def test():
    model.eval()
    te_correct = 0
    for batch_ind, (feats, targets, lens) in enumerate(te_loader):
        
        feats = pack_padded_sequence(feats, lens, 
                    batch_first=True, enforce_sorted=False)

        feats, targets = feats.to(device), targets.to(device)

        outputs = model.forward(feats)
        te_correct += corrects(outputs, targets)  

    print(f"Test Accuracy: {100 * te_correct/(len(te_loader.dataset)):.4f}%")

if __name__ == "__main__":

    for EPOCH_ID in range(200):
        tr_acc = train(EPOCH_ID)
        print(); print(f"Tra Accuracy: {tr_acc:.4f}%", end="\t")
        validate(EPOCH_ID)

    test()

    writer.flush()
    torch.save(model.state_dict(), "model_.pt")
