import numpy as np
import librosa
y, sr = librosa.load('kgbhup.mp3')
n_fft = 4096
full_spectrogram = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=n_fft/2, n_chroma = 12)
np.save('kgbhup.npy', full_spectrogram)
