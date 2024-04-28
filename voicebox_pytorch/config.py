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
        # CFM Wrapper
        "cfm_wrapper": {
            "cond_drop_prob": 0.2,
            "sigma": 1e-5,
        },
        # Model
        "model": {
            "dim": 512,
            "dim_cond_emb": 512,
            "num_cond_tokens": len(ipa_phonemes) + 25,
            "depth": 8,
            "dim_head": 64,
            "heads": 16,
        },
        # trainer
        "trainer": {
            "lr": 1e-4,
            "batch_size": 256,
            "num_train_steps": 1e6,
            "num_warmup_steps": 5000,
        },
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
