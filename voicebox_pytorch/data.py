import os
import json
import math
import textgrid
from tqdm import tqdm
from glob import glob
from pathlib import Path
from functools import wraps
from einops import rearrange
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Union

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


def exists(val):
    return val is not None


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


# dataset functions
def align_phoneme(labels, downsample_factor, sr=24000):
    sampling_rate = sr / downsample_factor
    silence_tokens = ["sp", "sil", "spn", ""]
    # Generate the sequence of labels based on duration and sampling rate
    label_sequence = []
    residual = 0
    for label, start_time, end_time in labels:
        if label in silence_tokens:
            label = "<SIL>"
        duration_in_seconds = end_time - start_time
        repeat_count = int(duration_in_seconds * sampling_rate)
        residual += duration_in_seconds * sampling_rate - repeat_count
        if residual > 1:
            repeat_count += 1
            residual -= 1
        label_sequence.extend([label] * repeat_count)
    end_total_time = math.ceil(end_time * sampling_rate)
    if len(label_sequence) < end_total_time - 2:
        label_sequence.extend(["<SIL>"] * ((end_total_time - 2) - len(label_sequence)))
    if len(label_sequence) >= end_total_time - 2:
        label_sequence = label_sequence[: end_total_time - 2]
    assert len(label_sequence) == end_total_time - 2, f"{len(label_sequence)} != {end_total_time - 2}"
    if label_sequence[-1] != "<SIL>":
        label_sequence[-1] = "<SIL>"
    return label_sequence


def parse_textgrid(file_path):
    tg = textgrid.TextGrid.fromFile(file_path)
    phonemes_with_timestamp = []
    for item in tg[1]:
        phonemes_with_timestamp.append((item.mark if item.mark else "<SIL>", item.minTime, item.maxTime))
    return phonemes_with_timestamp


class AudioDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder,
        audio_extension=".pt",
        phoneme_extension=".TextGrid",
        json_pathlist=None,
        tokenizer=None,
        reuturn_text=False,
        downsample_factor=None,
        split_to_use=["train-clean-100", "train-clean-360", "train-other-500"],
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), "folder does not exist"
        assert exists(tokenizer), "tokenizer must be provided"
        assert exists(json_pathlist), "json_pathlist must be provided"
        assert downsample_factor is not None, "downsample_factor must be provided"

        with open(json_pathlist, "r") as f:
            valid_files_per_split = json.load(f)
        # read file names for each split
        self.data = [
            {
                "audio": os.path.join(
                    folder,
                    split,
                    name.split("_")[0],
                    name.split("_")[1],
                    name + audio_extension,
                ),
                "textgrid": os.path.join(
                    folder,
                    split,
                    name.split("_")[0],
                    name.split("_")[1],
                    name + phoneme_extension,
                ),
            }
            for split, file_names in valid_files_per_split.items()
            if split in split_to_use
            for name in file_names
        ]

        self.tokenizer = tokenizer
        self.return_text = reuturn_text
        self.downsample_factor = downsample_factor
        self.audio_extension = audio_extension

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        phonemes = align_phoneme(parse_textgrid(data["textgrid"]), self.downsample_factor)
        phoneme_ids = self.tokenizer.encode_phonemes(phonemes)

        if self.audio_extension == ".pt":
            wave = torch.load(data["audio"])
        elif self.audio_extension == ".wav":
            wave, sr = torchaudio.load(data["audio"])
            assert sr == self.audio_encoder.sampling_rate, "sample rate must be 24_000"
            wave = rearrange(wave, "1 ... -> ...")

        seq_idx = -1 if self.audio_extension == ".wav" else 0
        if wave.size(seq_idx) > len(phoneme_ids):
            wave = wave[:, : len(phoneme_ids)] if self.audio_extension == ".wav" else wave[: len(phoneme_ids), :]

        data = {"wave": wave, "phoneme_ids": torch.tensor(phoneme_ids)}
        if self.return_text:
            text_file = data["textgrid"].replace(".TextGrid", ".original.txt")
            with open(text_file, "r") as f:
                text = f.read()
            data["text"] = text

        data["idx"] = idx

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
    return {"wave": audio, "phoneme_ids": phoneme_ids, "pad_mask": pad_mask, "idx": [datum["idx"] for datum in data]}


def get_dataloader(ds, pad_to_longest=True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)
