import json
from glob import glob
from pathlib import Path
from functools import wraps

from einops import rearrange
from tqdm import tqdm

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import torchaudio

# utilities


def exists(val):
    return val is not None


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


# dataset functions
def parse_lab_file(filename, sampling_rate=75.0):
    labels = []
    silence_tokens = ["sp", "sil", "spn", ""]
    with open(filename, "r") as file:
        for line in file:
            parts = line.split("\t")  # Strip and split on tabs

            start_time = float(parts[0])
            end_time = float(parts[1])
            label = parts[2].split("\n")[0]

            # Replace 'sp' with 'SIL' or add 'SIL' if the label is empty
            if label in silence_tokens:
                label = "<SIL>"

            labels.append((start_time, end_time, label))

    # Generate the sequence of labels based on duration and sampling rate
    label_sequence = []
    for start_time, end_time, label in labels:
        duration_in_seconds = end_time - start_time
        repeat_count = int(duration_in_seconds * sampling_rate)
        label_sequence.extend([label] * repeat_count)

    return label_sequence


class AudioDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder,
        audio_extension=".wav",
        phoneme_extension=".lab",
        json_pathlist=None,
        tokenizer=None,
        reuturn_text=False,
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), "folder does not exist"
        assert exists(tokenizer), "tokenizer must be provided"

        self.audio_extension = audio_extension
        phone_files = glob(f"{folder}/**/**/*{phoneme_extension}")
        print(phone_files[0])
        if not exists(json_pathlist):
            audio_files = [f.replace(phoneme_extension, audio_extension) for f in phone_files]
        else:
            with open(json_pathlist, "r") as f:
                pathlist = json.load(f)
            audio_files = [v["path"] for v in pathlist.values()]
            # intersect the two lists
            phone_files_names = [f.split(".")[0] for f in phone_files]
            audio_files = [f.split(".")[0] for f in audio_files]
            valid_files_names = set(phone_files_names) & set(audio_files)
            phone_files = [f + phoneme_extension for f in valid_files_names]
            audio_files = [f + audio_extension for f in valid_files_names]

        assert len(audio_files) > 0, "no audio_files found"
        assert len(audio_files) == len(phone_files), "audio_files and phone_files must have the same length"
        print(f"found {len(audio_files)} audio files and {len(phone_files)} phone files")

        audio_files.sort()
        phone_files.sort()

        self.data = list(zip(audio_files, phone_files))

        self.tokenizer = tokenizer
        self.return_text = reuturn_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_file, phone_file = self.data[idx]
        phonemes = parse_lab_file(phone_file)
        phoneme_ids = self.tokenizer.encode_phonemes(phonemes)

        wave, sr = torchaudio.load(audio_file)
        assert sr == 24_000, "sample rate must be 24_000"
        wave = rearrange(wave, "1 ... -> ...")

        data = {"wave": wave, "phoneme_ids": torch.tensor(phoneme_ids)}
        if self.return_text:
            text_file = phone_file.replace(".lab", ".original.txt")
            with open(text_file, "r") as f:
                text = f.read()
            data["text"] = text

        return data


# dataloader functions


def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = fn(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner


@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)


@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    audio = [datum["wave"] for datum in data]
    phoneme_ids = [datum["phoneme_ids"] for datum in data]
    audio = pad_sequence(audio, batch_first=True)
    phoneme_ids = pad_sequence(phoneme_ids, batch_first=True)
    pad_mask = ~(phoneme_ids == 0)
    return {"wave": audio, "phoneme_ids": phoneme_ids, "pad_mask": pad_mask}


def get_dataloader(ds, pad_to_longest=True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)
