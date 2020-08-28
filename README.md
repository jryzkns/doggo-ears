# doggo-ears

A naive program that listens to you and determines your mood when you speak to it. The model is trained on the [`RAVDESS`](https://smartlaboratory.org/ravdess/) and [`TESS`](https://tspace.library.utoronto.ca/handle/1807/24487) datasets and features a many-to-one LSTM classifier.

## Prerequisites

- Python 3.6.X, the program is tested and developed in
- PyTorch, which can be installed with the following [command](https://stackoverflow.com/questions/56859803/modulenotfounderror-no-module-named-tools-nnwrap/56877423#56877423):
```py
pip3 install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-win_amd64.whl
```
- PyAudio, install with the following [instructions](https://stackoverflow.com/a/55630212/12275558)
- librosa
- [OPTIONAL] `tensorboard` for visualizing training progress