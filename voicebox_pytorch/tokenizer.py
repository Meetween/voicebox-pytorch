import torch
import sentencepiece as spm


class Tokenizer:
    def __init__(self, config):
        self.vocab_size = config.tokenizer.vocab_size

        # Tokens
        self.pad_token = config.tokenizer.pad_token
        self.silence_token = config.tokenizer.silence_token
        self.sequence_begin_token = config.tokenizer.sequence_begin_token
        self.sequence_end_token = config.tokenizer.sequence_end_token
        self.unknown_token = config.tokenizer.unknown_token

        # Phoneme map
        self.phoneme_to_id = {}
        self.id_to_phoneme = {}
        for p in range(len(config.tokenizer.vocab_output)):
            self.phoneme_to_id[config.tokenizer.vocab_output[p]] = p
            self.id_to_phoneme[p] = config.tokenizer.vocab_output[p]

        # IDs
        self.silence_token_id = self.phoneme_to_id[self.silence_token]
        self.sequence_begin_token_id = self.phoneme_to_id[self.sequence_begin_token]
        self.sequence_end_token_id = self.phoneme_to_id[self.sequence_end_token]
        self.unknown_token_id = self.phoneme_to_id[self.unknown_token]

    def encode_phonemes(self, phonemes):
        return (
            [self.sequence_begin_token_id] + [self.phoneme_to_id[p] for p in phonemes] + [self.sequence_end_token_id]
        )

    def decode_phonemes(self, phonemes):
        return [self.id_to_phoneme[p] for p in phonemes]
