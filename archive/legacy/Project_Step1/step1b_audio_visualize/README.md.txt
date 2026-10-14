# Step 1B: Audio Visualization

## Overview
This script creates visual comparisons between the original raw audio and the processed audio from Step 1A.

## What It Does
1. **Waveform Plotting:** Displays the time-domain graphs of the audio before and after processing, showing how the silence was trimmed and the signal was centered.
2. **Spectrogram Generation:** Displays the frequency-domain (Mel-spectrogram) graphs to show the acoustic energy of the speech over time.

## Why We Need It
We must ensure that our cleaning process does not destroy any important vocal features. These visualizations act as a quick "sanity check" to visually confirm that the core clinical speech data remains perfectly intact after preprocessing.