from typing import List
from .arpa_to_ipa import arpa_to_ipa_lookup


class Tokenizer:
    def __init__(self, config):
        self.vocab_size = config.tokenizer.vocab_size
        self.arpa_to_ipa_lookup = arpa_to_ipa_lookup

        # Tokens
        self.pad_token = config.tokenizer.pad_token
        self.silence_token = config.tokenizer.silence_token
        self.sequence_begin_token = config.tokenizer.sequence_begin_token
        self.sequence_end_token = config.tokenizer.sequence_end_token
        self.unknown_token = config.tokenizer.unknown_token
        self.special_tokens = config.tokenizer.special_tokens

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

    def arpa_to_ipa(self, arpa_phonemes: List[str]):
        ipa_phonemes = []
        for p in arpa_phonemes:
            if p not in self.special_tokens:
                try:
                    ipa_phonemes.append(self.arpa_to_ipa_lookup[p])
                except KeyError:
                    raise ValueError(f"Phoneme {p} not found in the arpa_to_ipa_lookup")
            else:  # special tokens handling
                ipa_phonemes.append(p)
        return ipa_phonemes

    def encode_phonemes(self, phonemes, convert_to_ipa=True):
        return (
            [self.sequence_begin_token_id] + [self.phoneme_to_id[p] for p in self.arpa_to_ipa(phonemes)]
            if convert_to_ipa
            else [self.phoneme_to_id[p] for p in phonemes] + [self.sequence_end_token_id]
        )

    def decode_phonemes(self, phonemes):
        return [self.id_to_phoneme[p] for p in phonemes]
