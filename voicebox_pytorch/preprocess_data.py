import os
import shutil
import json
from glob import glob
import argparse
import librosa
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Preprocess the dataset")
parser.add_argument(
    "--data_dir", type=str, default="$SCRATCH/data/LibriTTS/train-clean-360", help="path to the dataset"
)
parser.add_argument(
    "--phonemes_dir",
    type=str,
    default="$PLG_GROUPS_STORAGE/plggmeetween/meetween_datasets/LibriTTSCorpusLabel/lab/phone/train-clean-360",
    help="path to the integrated phoneme labels",
)

"""This script takes the folder of the original dataset and integrates phoneme labels information, providing an hdf5 file"""

if __name__ == "__main__":
    args = parser.parse_args()
    SCRATCH = os.environ.get("SCRATCH")
    PLG_GROUPS_STORAGE = os.environ.get("PLG_GROUPS_STORAGE")
    data_dir = args.data_dir.replace("$SCRATCH", SCRATCH)
    phonemes_dir = args.phonemes_dir.replace("$PLG_GROUPS_STORAGE", PLG_GROUPS_STORAGE)
    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"
    assert os.path.exists(phonemes_dir), f"Phonemes directory {phonemes_dir} does not exist"

    # create a json file with the audio file that are below 5s lenghts

    audio_files = glob(os.path.join(data_dir, "**", "**", "*.wav"))
    print(f"Found {len(audio_files)} audio files")
    valid_audio_files = {}
    for audio_file in tqdm(audio_files):
        duration = librosa.get_duration(filename=audio_file)
        if duration < 8:
            phone_file = audio_file.replace(".wav", ".lab")
            if not os.path.exists(phone_file):
                print(f"Phoneme file {phone_file} does not exist")
                continue
            file_name = audio_file.replace(".wav", "").replace(data_dir, "")
            valid_audio_files[file_name] = {"duration": duration, "path": audio_file}
    print(f"Found {len(valid_audio_files)} audio files below 8s")
    with open("valid_audio_files.json", "w") as f:
        json.dump(valid_audio_files, f)

    # # Iterate over the dataset and integrate the phoneme labels
    # lab_files = glob(os.path.join(phonemes_dir, "**", "**", "*.lab"))
    # print(f"Found {len(lab_files)} lab files")
    # for lab_file in tqdm(lab_files):
    #     # Get the corresponding wav file
    #     dest_lab_file = lab_file.replace(
    #         f"{PLG_GROUPS_STORAGE}/plggmeetween/meetween_datasets/LibriTTSCorpusLabel/lab/phone/train-clean-360/",
    #         f"{SCRATCH}/data/LibriTTS/train-clean-360/",
    #     )
    #     wav_file = dest_lab_file.replace(".lab", ".wav")
    #     if os.path.exists(wav_file):
    #         # Copy the lab file to the same directory as the wav file
    #         shutil.copy(lab_file, dest_lab_file)
