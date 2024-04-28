import torch
import random
import argparse
import datetime
import torchaudio
import random
import glob
from tqdm import tqdm
from data import get_dataloader
from config import config
from einops import rearrange
from data import AudioDataset
from tokenizer.tokenizer import Tokenizer
from tensorboardX import SummaryWriter
from voicebox_pytorch import VoiceBox, EncodecVoco, ConditionalFlowMatcherWrapper

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="results/voicebox.74000.pt")
parser.add_argument(
    "--audio_path",
    type=str,
    default="/home/ubuntu/data/LibriTTSR/LibriTTS_R",
)


def cycle(dl):
    while True:
        for data in dl:
            yield data


if __name__ == "__main__":
    """Train example without text_to_semantic SpearTTS model."""

    log_dir = "./inference_runs"
    existing_runs = glob.glob(f"{log_dir}/*")
    run_index = len(existing_runs) + 1
    writer = SummaryWriter(log_dir=f"{log_dir}/{run_index}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    # phonem tokenizer
    tokenizer = Tokenizer(config)

    # audio encoder
    audio_enc_dec = EncodecVoco()
    downsample_factor = audio_enc_dec.downsample_factor
    # dataset
    dataset = AudioDataset(
        folder=args.audio_path,
        json_pathlist="test_files.json",
        tokenizer=tokenizer,
        downsample_factor=downsample_factor,
        audio_extension=".pt",
        split_to_use=["test-other", "test-clean"],
    )
    dl_iter = cycle(get_dataloader(dataset, batch_size=16, shuffle=True, drop_last=True))

    # prepare cfm wrapper
    model = VoiceBox(
        dim=512,
        dim_cond_emb=512,
        audio_enc_dec=audio_enc_dec,
        num_cond_tokens=tokenizer.vocab_size + 20,  # number of phonemes + special tokens
        depth=12,
        dim_head=64,
        heads=16,
        ff_mult=4,
        attn_qk_norm=False,
        num_register_tokens=0,
        use_gateloop_layers=False,
    )

    cfm_wrapper = ConditionalFlowMatcherWrapper(voicebox=model, cond_drop_prob=0.2)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    cfm_wrapper.load_state_dict(checkpoint["model"])
    cfm_wrapper = cfm_wrapper.to(device)

    cond_scale = 1.3
    num_steps = 32
    steps = args.checkpoint_path.split(".")[-2]

    with torch.inference_mode():
        (batch,) = next(dl_iter)

        # select 8 example for unconditional and 8 for conditional
        batch_unconditional = {k: v[:8] for k, v in batch.items()}
        batch_conditional = {k: v[8:16] for k, v in batch.items()}

        # get the idx of the files
        idx_unconditional = batch_unconditional["idx"]
        idx_conditional = batch_conditional["idx"]
        print(f"Unconditional idx: {idx_unconditional}")
        print(f"Conditional idx: {idx_conditional}")

        # for cond_scale in tqdm([1.0, 1.3, 1.6]):
        print(f"Generating {cond_scale} cond_scale")
        # unconditional generation
        wave_cond = batch_unconditional["wave"].to(device)
        phoneme_ids_cond = batch_unconditional["phoneme_ids"].to(device)
        mask_cond = batch_unconditional["pad_mask"].to(device)

        output_waves = cfm_wrapper.sample(
            phoneme_ids=phoneme_ids_cond,
            steps=num_steps,
            cond_scale=cond_scale,
        )

        # save audio
        for i, wave in enumerate(output_waves):
            first_false_index = torch.argmax((~mask_cond[i]).to(torch.float32)) * 320
            # truncate the wave following the pad maskf
            wave = wave[:, : int(first_false_index)]
            try:
                writer.add_audio(
                    f"Unconditional/sample_{i}",
                    wave.detach().cpu().view(-1).unsqueeze(-1),
                    steps,
                    sample_rate=24_000,
                )
            except Exception as e:
                print("Uncond error", e)
                pass

            # upload text
            file_wav = dataset.data[idx_unconditional[i]]["audio"]
            with open(file_wav.replace(".pt", ".original.txt"), "r") as f:
                text = f.read()
            writer.add_text(f"Unconditional/sample_{i}", text, steps)

        # batch conditional generation
        # get 3s of the conditioning wave
        wave_cond = wave_cond[:, : int(5 * 75)]
        phoneme_ids_cond = phoneme_ids_cond[:, : int(5 * 75)]

        wave = batch_conditional["wave"].to(device)
        phoneme_ids = batch_conditional["phoneme_ids"].to(device)
        mask = batch_conditional["pad_mask"].to(device)
        wave_seq_len = wave.size(1)  # already encoded

        condition = torch.cat([wave_cond, wave], dim=1).to(torch.float32)  # assuming encodec audio
        condition_mask = torch.cat(
            [
                torch.zeros_like(phoneme_ids_cond, dtype=torch.bool),
                torch.ones_like(phoneme_ids, dtype=torch.bool),
            ],
            dim=-1,
        ).to(device)
        phoneme_ids = torch.cat([phoneme_ids_cond, phoneme_ids], dim=-1)

        output_waves = cfm_wrapper.sample(
            phoneme_ids=phoneme_ids,
            cond=condition,
            cond_mask=condition_mask,
            steps=num_steps,
            cond_scale=cond_scale,
        )

        # save audio
        for i, wave in enumerate(output_waves):
            first_false_index = torch.argmax((~mask[i]).to(torch.float32)) * 320 + (
                wave_cond.size(1) * 320
            )  # assuming encodec audio
            original_wave = wave[:, : int(wave_cond.size(1) * 320)]
            # truncate the wave following the pad mask
            wave = wave[:, int(wave_cond.size(1) * 320) : int(first_false_index)]
            try:
                writer.add_audio(
                    f"Conditional/sample_{i}",
                    wave.detach().cpu().view(-1).unsqueeze(-1),
                    steps,
                    sample_rate=24_000,
                )
            except Exception as e:
                print("Conditional error", e)
                pass

            try:
                writer.add_audio(
                    f"Original_condition/sample_{i}",
                    original_wave.detach().cpu().view(-1).unsqueeze(-1),
                    steps,
                    sample_rate=24_000,
                )
            except Exception as e:
                print("Original error", e)
                pass

            # upload text
            file_wav = dataset.data[idx_conditional[i]]["audio"]
            with open(file_wav.replace(".pt", ".original.txt"), "r") as f:
                text = f.read()
            writer.add_text(f"Conditional/sample_{i}", text, steps)
