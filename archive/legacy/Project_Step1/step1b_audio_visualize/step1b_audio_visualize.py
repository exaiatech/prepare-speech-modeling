import matplotlib.pyplot as plt
import librosa
import numpy as np

plt.figure(figsize=(14, 10))

#1. Original Sound Waveform
plt.subplot(2, 2, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.title('Original Sound Wave')

# 2. Processed Sound Waveform
plt.subplot(2, 2, 2)
librosa.display.waveshow(y_trimmed, sr=sr, alpha=0.5, color='orange')
plt.title('Processed Sound Wave')

#3. Original Spectrogram
plt.subplot(2, 2, 3)
S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(S_orig, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Original Spektrogram')

#4. Processed Spectrogram
plt.subplot(2, 2, 4)
S_proc = librosa.amplitude_to_db(np.abs(librosa.stft(y_trimmed)), ref=np.max)
librosa.display.specshow(S_proc, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Processed Spektrogram')

plt.tight_layout()
plt.show()

