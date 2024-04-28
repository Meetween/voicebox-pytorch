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

from collections import defaultdict

"""This script takes the folder of the original dataset and integrates phoneme labels information, providing an hdf5 file"""

parser = argparse.ArgumentParser()
parser.add_argument("--split_index", type=int, required=True)
parser.add_argument("--normalize", action="store_true")

if __name__ == "__main__":
    # parse
    args = parser.parse_args()

    weight_dtype = torch.float16
    normalize = args.normalize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_encoder = EncodecVoco()
    audio_encoder = audio_encoder.to(device).to(weight_dtype)

    with open("valid_short_files.json", "r") as f:
        valid_files = json.load(f)
    root = "/home/ubuntu/data/LibriTTSR/LibriTTS_R"

    all_files = []
    for split, files in valid_files.items():
        if "train" in split:
            continue
        all_files += [(split, file) for file in files]

    # split files in 8 chunks and select the chunk of the split_index
    n_chunks = 8
    chunk_size = len(all_files) // n_chunks
    start = chunk_size * args.split_index
    end = chunk_size * (args.split_index + 1)
    if args.split_index == n_chunks - 1:
        end = len(all_files)

    valid_chunked_file = defaultdict(lambda: [])
    for split, f in all_files[start:end]:
        valid_chunked_file[split].append(f)

    for split, files in valid_chunked_file.items():
        print(f"Split: {split} - {len(files)} files")
        for file in tqdm(files, desc=f"Processing {split} files"):
            speaker, folder = file.split("_")[:2]
            audio_file = os.path.join(root, split, speaker, folder, f"{file}.wav")
            # encode file with encoder and save tensor
            dest_file_name = os.path.join(root, split, speaker, folder, f"{file}.pt")
            wave, sr = torchaudio.load(audio_file)
            assert sr == audio_encoder.sampling_rate, "sample rate must be 24_000"
            wave = rearrange(wave, "1 ... -> ...")
            wave_encoded = audio_encoder.encode(wave.to(device).to(weight_dtype))
            torch.save(wave_encoded.detach().cpu(), dest_file_name)
