import numpy as np
from scipy import stats

#1. SNR (Signal-to-Noise Ratio) Calculation Function
def calculate_snr(signal):
    # Overall strength of the signal
    signal_power = np.mean(signal**2)
    # We use the lowest 5% of the signal to estimate noise level.
    noise_power = np.mean(np.sort(signal**2)[:int(len(signal)*0.05)])
    
    # Safety precaution in case of a division by zero error.
    if noise_power == 0:
        return float('inf')
    
    # Formula: 10 * log10(Signal_Power / Noise_Power)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

snr_orijinal = calculate_snr(y)
snr_proc = calculate_snr(y_trimmed)

print(snr_orijinal)
print(snr_proc)

#2. Kolmogorov-Smirnov (K-S) Test
y_orig_norm = librosa.util.normalize(y)
ks_stat, p_value =stats.ks_2samp(y_orig_norm, y_trimmed)

print(ks_stat)
print(p_value)
     


