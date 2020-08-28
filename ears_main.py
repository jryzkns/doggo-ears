from doggo_ears_definitions import *

import pyaudio

from model import classifier
from utils import generate_aud_features, argmax

STATEDICT_PATH = "model_84513.pt"
m_ = classifier(N_FEATURES, N_CATEGORIES)
m_.load_state_dict(torch.load(STATEDICT_PATH))

global aud_buffer, features
aud_buffer, features = np.ndarray((0,)), []

def on_receive_audio(in_data, frame_count, time_info, flag):

    global aud_buffer, features, preds
    aud_buffer = np.concatenate((aud_buffer, np.fromstring(in_data, dtype=np.float32)))

    if aud_buffer.shape[0] > samples_per_wind:
        features += [generate_aud_features(aud_buffer[:samples_per_wind])]
        aud_buffer = aud_buffer[samples_per_wind>>1:]

    if len(features) > NUM_INFERENCE_WINDOW:
        preds = m_.pred(features[:NUM_INFERENCE_WINDOW])
        print(preds, emote_idn[argmax(preds)])
        features = features[NUM_INFERENCE_WINDOW//3:]

    return in_data, pyaudio.paContinue
    
def run_ears():

    p = pyaudio.PyAudio()
    dog_ears = p.open(  format=pyaudio.paFloat32,
                        channels=1, rate=LISTENER_RATE,
                        output=False, input=True,
                        stream_callback=on_receive_audio)

    dog_ears.start_stream()
    while dog_ears.is_active():
        continue
        
    dog_ears.close()
    p.terminate()

if __name__ == "__main__":
    run_ears()