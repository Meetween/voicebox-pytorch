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
parser.add_argument("--checkpoint_path", type=str, default="results/voicebox.4000.pt")
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
    # audio encoder
    audio_enc_dec = EncodecVoco()
    downsample_factor = audio_enc_dec.downsample_factor
    # dataset
    dataset = AudioDataset(
        folder=args.audio_path,
        json_pathlist="valid_short_files.json",
        tokenizer=tokenizer,
        downsample_factor=downsample_factor,
    )

    # prepare cfm wrapper
    model = VoiceBox(
        dim=512,
        dim_cond_emb=512,
        audio_enc_dec=EncodecVoco(),
        num_cond_tokens=tokenizer.vocab_size + 20,  # number of phonemes + special tokens
        depth=8,
        dim_head=64,
        heads=16,
        ff_mult=4,
        attn_qk_norm=False,
        num_register_tokens=0,
        use_gateloop_layers=False,
    )

    cfm_wrapper = ConditionalFlowMatcherWrapper(voicebox=model)

    # Let's train!
    trainer = VoiceBoxTrainer(
        cfm_wrapper=cfm_wrapper,
        dataset=dataset,
        lr=1e-4,
        batch_size=256,
        num_train_steps=1e6,
        accelerator=accelerator,
    )
    if args.resume_training:
        trainer.load(args.checkpoint_path)
    trainer.train()
