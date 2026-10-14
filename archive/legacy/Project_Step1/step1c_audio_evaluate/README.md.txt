# Step 1C: Statistical Audio Evaluation

## Overview
This script runs mathematical and statistical tests to measure the quality of our audio processing.

## What It Does
1. **SNR (Signal-to-Noise Ratio) Calculation:** Calculates the SNR for both the original and the processed audio to measure how much the background noise was reduced.
2. **Kolmogorov-Smirnov (K-S) Test:** Runs a statistical test comparing the distribution of the original audio against the processed audio.

## Why We Need It
In clinical linguistics, we cannot rely only on our ears or eyes; we need numbers. The SNR proves that the audio is clearer, while the K-S test mathematically proves that the fundamental characteristics of the patient's voice were not corrupted during the cleaning phase.