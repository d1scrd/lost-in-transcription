import os
import librosa
import matplotlib.pyplot as plt

#* Simple waveform plotter
AUDIO_FILE = "../audio_kdd/Besigheid/Besigheid1.m4av"
OUTPUT_DIR = "waveforms"
FIG_SIZE = (12, 4)

#* Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output folder: '{OUTPUT_DIR}'")

#* Prepare image path
base_name = os.path.splitext(os.path.basename(AUDIO_FILE))[0]
waveform_image = os.path.join(OUTPUT_DIR, f"{base_name}_waveform.png")

#* Load audio
print(f"Loading audio file '{AUDIO_FILE}'...")
y, sr = librosa.load(AUDIO_FILE, sr=None)
print(f"Loaded {len(y)} samples at {sr} Hz")

#* Plot the waveform
plt.figure(figsize=FIG_SIZE)
librosa.display.waveshow(y, sr=sr, color='mediumblue')
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig(waveform_image)
plt.close()

print(f"Saved waveform image to '{waveform_image}'")
