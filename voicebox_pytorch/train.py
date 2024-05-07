import argparse
import torch
from config import config
from trainer import VoiceBoxTrainer
from tokenizer.tokenizer import Tokenizer

from data import AudioDataset
from voicebox_pytorch import (
    VoiceBox,
    EncodecVoco,
    ConditionalFlowMatcherWrapper,
)
from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="results/voicebox.11000.pt")
parser.add_argument("--resume_training", action="store_true")
parser.add_argument(
    "--audio_path",
    type=str,
    default="/home/ubuntu/data/LibriTTSR/LibriTTS_R",
)

if __name__ == "__main__":
    """Train example without text_to_semantic SpearTTS model."""
    args = parser.parse_args()

    # set up the accelerator

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    split_batches = (False,)
    accelerate_kwargs = {}

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        split_batches=split_batches,
        mixed_precision="fp16",
        **accelerate_kwargs,
    )
    # phonem tokenizer
    tokenizer = Tokenizer(config)
    assert tokenizer.pad_token_id == 0, "pad token id must be 0"
    # audio encoder
    audio_enc_dec = EncodecVoco()
    downsample_factor = audio_enc_dec.downsample_factor
    # dataset
    dataset = AudioDataset(
        folder=args.audio_path,
        json_pathlist="train_ids.json",
        tokenizer=tokenizer,
        downsample_factor=downsample_factor,
        audio_extension=".wav",
    )

    # prepare cfm wrapper
    model = VoiceBox(
        dim=512,
        dim_cond_emb=512,
        audio_enc_dec=EncodecVoco(bandwidth_id=2),
        num_cond_tokens=tokenizer.vocab_size + 20,  # number of phonemes + special tokens + extra tokens
        depth=12,
        dim_head=64,
        heads=16,
        ff_mult=4,
        attn_qk_norm=False,
        num_register_tokens=0,
        use_gateloop_layers=False,
    )

    cfm_wrapper = ConditionalFlowMatcherWrapper(
        voicebox=model,
        cond_drop_prob=0.2,
        sigma=1e-5,
    )

    # Let's train!
    trainer = VoiceBoxTrainer(
        cfm_wrapper=cfm_wrapper,
        dataset=dataset,
        lr=1e-4,
        batch_size=128,
        num_train_steps=150_000,
        num_warmup_steps=5000,
        accelerator=accelerator,
        save_results_every=1000,
        save_model_every=1000,
    )
    if args.resume_training:
        trainer.load(args.checkpoint_path)
    trainer.train()
