from doggo_ears_definitions import *
from multiprocessing import get_context
from utils import generate_aud_features

import librosa
import pickle
import os

def add_data_job(payload):
    
    fn, cat = payload
    samples, rate = librosa.load(fn, res_type='kaiser_fast')

    samples_per_wind = int(rate * window_duration)
    start, end = 0, samples_per_wind

    features = []
    while end < len(samples) - 1:
        features += [generate_aud_features(samples[start:end])]
        start += (samples_per_wind >> 1)
        end = end + (samples_per_wind >> 1) \
                if end + (samples_per_wind >> 1) < len(samples) \
                else len(samples) - 1

    return np.asarray(features, dtype=np.single), cat

if __name__ == "__main__":

    # generate datapoint paths
    RAVDESS_PTRS, TESS_PTRS = \
        [os.path.join(RAVDESS_DSET_PATH, adir_, dpfn_)
            for adir_ in os.listdir(RAVDESS_DSET_PATH)
            for dpfn_ in os.listdir(os.path.join(RAVDESS_DSET_PATH, adir_))], \
        [os.path.join(TESS_DSET_PATH, dpfn_)
                for dpfn_ in os.listdir(TESS_DSET_PATH)
                if dpfn_.endswith(".wav")]

    # filter out unwanted datapoints and couple path with label
    RAVDESS_PTRS, TESS_PTRS = \
        [ (path_, emote_id[path_.split("-")[-5]]) for path_ in RAVDESS_PTRS 
            if path_.split("-")[-5] in emote_id.keys()], \
        [ (path_, path_.split("_")[-1][:-4]) for path_ in TESS_PTRS
            if path_.split("_")[-1][:-4] in emote_id.values()]
    
    dataset = {emotion : [] for emotion in emote_id.values()}
    with get_context("spawn").Pool(processes = N_WORKERS) as pooler:
        pooler.daemon = True
        for d, c in pooler.map( add_data_job, RAVDESS_PTRS + TESS_PTRS):
            dataset[c] += [d]

    with open("dataset.pkl", "wb") as outf_:
        pickle.dump(tuple([(label, datapoint)
                            for label, data in dataset.items() 
                            for datapoint in data]), outf_)