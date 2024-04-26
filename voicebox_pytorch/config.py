from tokenizer.arpa_to_ipa import arpa_to_ipa_lookup


def dict_to_object(src):
    class DictToObject:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    value = DictToObject(value)
                self.__dict__[key] = value

        def __repr__(self):
            return f"{self.__dict__}"

    return DictToObject(src)


ipa_phonemes = list(arpa_to_ipa_lookup.values())
config = dict_to_object(
    {
        # Audio
        "audio": {
            # "token_duration": 256 / 24000 # 256 samples at 24kHz
            "token_duration": 0.01,  # To match the 100Hz token duration or Montreal Forced Aligner
            "sample_rate": 24000,
            "hop_size": 256,
        },
        # Architecture
        "gpt": {
            "n_embeddings": 512,
            "n_heads": 16,
            "n_layers": 16,
            "n_dim": 512,
            "n_dim_head": 32,  # n_dim / n_heads
            "n_dim_ffn": 32 * 4,  # 4 * n_dim_head
            # # n_dim_duration + n_dim_pitch should be less than n_dim
            # "n_dim_duration": 128,
            # "n_dim_pitch": 128,
            # Maximum duration values
            "max_duration": 100,
        },
        "tokenizer_style": {"pitch_min": -2, "pitch_max": 2, "tokens": 256},
        # Tokenizer
        "tokenizer": {
            "vocab_size": 5 + len(ipa_phonemes),
            "vocab_output": [
                "<PAD>",
                "<BOS>",
                "<EOS>",
                "<SIL>",
                "<UNK>",
            ]
            + ipa_phonemes,
            # Special tokens
            "pad_token": "<PAD>",
            "silence_token": "<SIL>",
            "sequence_begin_token": "<BOS>",
            "sequence_end_token": "<EOS>",
            "unknown_token": "<UNK>",
            "special_tokens": ["<PAD>", "<SIL>", "<BOS>", "<EOS>", "<UNK>"],
        },
    }
)
