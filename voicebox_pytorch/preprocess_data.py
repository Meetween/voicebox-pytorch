import os
import json
import librosa
import argparse
from tqdm import tqdm
from glob import glob
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--audio_extension", type=str, default="wav")
parser.add_argument("--alignement_extension", type=str, default="TextGrid")
parser.add_argument("--max_duration", type=float, default=10.0)
parser.add_argument("--root", type=str, default="/home/ubuntu/data/LibriTTSR/LibriTTS_R")

"""This script takes the folder of the original dataset and integrates phoneme labels information, providing an hdf5 file"""

if __name__ == "__main__":
    # parse
    args = parser.parse_args()
    root = args.root
    # check all the files having a .pt and a .TextGrid extension
    # root, split, speaker, folder, file
    all_audio_files = [f.split(".")[0] for f in glob(f"{root}/**/**/**/*.{args.audio_extension}")]
    print(f"Found {len(all_audio_files)} audio files")
    all_alignement_files = [f.split(".")[0] for f in glob(f"{root}/**/**/**/*.{args.alignement_extension}")]
    valid_files_name = list(set(all_audio_files) & set(all_alignement_files))
    print(f"Found {len(valid_files_name)} valid files")
    valid_files = defaultdict(lambda: [])
    valid_files_name = [f.replace(f"{root}/", "") for f in valid_files_name]
    for file in valid_files_name:
        split, speaker, folder, file = file.split("/")
        valid_files[split].append(file)

    for split, files in valid_files.items():
        print(f"Split: {split} - {len(files)} files")

    # create a json file with the audio file that are below 10s lenghts
    valid_audio_files = {}
    for split, files in valid_files.items():
        if not "test" in split:
            continue
        valid_audio_files[split] = []
        print(f"Split: {split} - {len(files)} files")
        for file in tqdm(files, desc=f"Processing {split} files"):
            speaker, folder = file.split("_")[:2]
            try:
                audio_file = os.path.join(
                    root, split, speaker, folder, f"{file}.wav"
                )  # assuming wav present also with .pt as audio extension
                duration = librosa.get_duration(filename=audio_file)
            except Exception as e:
                print(f"Error: {e}")
                continue

            if duration <= args.max_duration and duration >= 1:
                valid_audio_files[split].append(file)

    with open("test_ids.json", "w") as f:
        json.dump(valid_audio_files, f)
