import numpy as np

import torch
torch.manual_seed(0)

# PRE-PROCESSING
RAVDESS_DSET_PATH = "C:\\Users\\***\\Downloads\\RAVDESS\\"
TESS_DSET_PATH = "C:\\Users\\***\\Downloads\\TESS\\"
N_WORKERS = 15

# DATASET
emote_id = {    
    "01" : "neutral",   "03" : "happy",
    "04" : "sad",       "05" : "angry"}

emote_idn = {
    0 : "neutral",  1 : "happy",
    2 : "sad",      3 : "angry"}

N_CATEGORIES = len(emote_id)
label_id = { n : torch.tensor(i)
    for i, n in enumerate(emote_id.values())}

# AUDIO
window_duration = 0.5
LISTENER_RATE = 44100
N_FEATURES = 2
NUM_INFERENCE_WINDOW = 10
samples_per_wind = int(LISTENER_RATE * window_duration)

# TRAINING
BATCH_SIZE = 16
loader_params = {   "batch_size" : BATCH_SIZE,
                    "shuffle" : True}