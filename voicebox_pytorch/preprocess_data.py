import os
import shutil
import json
from glob import glob
import argparse
import librosa
from tqdm import tqdm
import torch
from einops import rearrange
import torchaudio
from voicebox_pytorch import (
    VoiceBox,
    EncodecVoco,
    ConditionalFlowMatcherWrapper,
)

"""This script takes the folder of the original dataset and integrates phoneme labels information, providing an hdf5 file"""

if __name__ == "__main__":
    with open("valid_files.json", "r") as f:
        valid_files = json.load(f)
    # create a json file with the audio file that are below 5s lenghts
    root = "/home/ubuntu/data/LibriTTSR/LibriTTS_R"
    valid_audio_files = {}
    for split, files in valid_files.items():
        valid_audio_files[split] = []
        print(f"Split: {split} - {len(files)} files")
        for file in tqdm(files, desc=f"Processing {split} files"):
            speaker, folder = file.split("_")[:2]
            try:
                audio_file = os.path.join(root, split, speaker, folder, f"{file}.wav")
                duration = librosa.get_duration(filename=audio_file)
            except Exception as e:
                print(f"Error: {e}")
                continue

            if duration <= 10:
                valid_audio_files[split].append(file)

    with open("valid_short_files.json", "w") as f:
        json.dump(valid_audio_files, f)
