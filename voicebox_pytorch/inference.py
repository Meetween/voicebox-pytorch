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
parser.add_argument("--checkpoint_path", type=str, default="results/voicebox.156000.pt")
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

    torch.manual_seed(42)

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
        json_pathlist="test_ids.json",
        tokenizer=tokenizer,
        downsample_factor=downsample_factor,
        audio_extension=".wav",
        split_to_use=["test-other", "test-clean"],
    )

    speed_dataset = AudioDataset(
        folder=args.audio_path,
        json_pathlist="test_ids.json",
        tokenizer=tokenizer,
        downsample_factor=downsample_factor,
        audio_extension=".wav",
        split_to_use=["test-other", "test-clean"],
        speed_factor=0.8,
    )

    slow_dataset = AudioDataset(
        folder=args.audio_path,
        json_pathlist="test_ids.json",
        tokenizer=tokenizer,
        downsample_factor=downsample_factor,
        audio_extension=".wav",
        split_to_use=["test-other", "test-clean"],
        speed_factor=1.25,
    )
    # FIXME
    # assuming data are already shuffled in the dataset class
    dl_iter = cycle(get_dataloader(dataset, batch_size=16, shuffle=False, drop_last=False))
    dl_iter_speed = cycle(get_dataloader(speed_dataset, batch_size=16, shuffle=False, drop_last=False))
    dl_iter_slow = cycle(get_dataloader(slow_dataset, batch_size=16, shuffle=False, drop_last=False))

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
        for name, iterable in zip(["1x"], [dl_iter]): #["1x", "1.25x", "0.8x"], [dl_iter, dl_iter_speed, dl_iter_slow]
            (batch,) = next(iterable)

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
            original_voice = batch_unconditional["wave"].to(device)
            # encode to codes
            original_voice_encoded = audio_enc_dec.encode(original_voice.to(device)).squeeze(0)
            original_phonemes = batch_unconditional["phoneme_ids"].to(device)
            mask_original = batch_unconditional["pad_mask"].to(device)

            output_waves = cfm_wrapper.sample(
                phoneme_ids=original_phonemes,
                steps=num_steps,
                cond_scale=cond_scale,
            )

            # save audio
            for i, wave in enumerate(output_waves):
                first_false_index = torch.argmax((~mask_original[i]).to(torch.float32)) * 320
                # truncate the wave following the pad maskf
                wave = wave[:, : int(first_false_index)]
                try:
                    writer.add_audio(
                        f"Unconditional_{name}/sample_{i}",
                        wave.detach().cpu().view(-1).unsqueeze(-1),
                        steps,
                        sample_rate=24_000,
                    )
                except Exception as e:
                    print("Uncond error", e)
                    pass

                # upload text
                if name == "1x":
                    file_wav = dataset.data[idx_unconditional[i]]["audio"]
                    with open(file_wav.replace(".wav", ".original.txt"), "r") as f:
                        text = f.read()
                    writer.add_text(f"Unconditional/sample_{i}", text, steps)
                # upload audio
                    writer.add_audio(
                        f"Voice_infilled/sample_{i}",
                        original_voice[i].detach().cpu().view(-1).unsqueeze(-1),
                        steps,
                        sample_rate=24_000,
                    )
                


            # batch conditional generation
            # get 3s of the conditioning wave
            # 0
            #original_voice = original_voice
            #original_phonemes = original_phonemes
             # 1
            conditioning_voice = batch_conditional["wave"].to(device)
            conditioning_voice_encoded = audio_enc_dec.encode(conditioning_voice.to(device)).squeeze(0)[:, : int(3 * 75)]
            phoneme_ids = batch_conditional["phoneme_ids"].to(device)[:, : int(3 * 75)]
            mask = batch_conditional["pad_mask"].to(device)
            wave_seq_len = conditioning_voice.size(1)  # already encoded

            condition = torch.cat([conditioning_voice_encoded, torch.randn_like(original_voice_encoded)], dim=1).to(torch.float32)  # assuming encodec audio
            condition_mask = torch.cat(
                [
                    torch.zeros_like(phoneme_ids, dtype=torch.bool), 
                    torch.ones_like(original_phonemes, dtype=torch.bool),
                ],
                dim=-1,
            ).to(device)
            phoneme_ids = torch.cat([phoneme_ids, original_phonemes], dim=-1)

            output_waves = cfm_wrapper.sample(
                phoneme_ids=phoneme_ids,
                cond=condition,
                cond_mask=condition_mask,
                steps=num_steps,
                cond_scale=cond_scale,
            )

            # save audio
            for i, wave in enumerate(output_waves):
                first_false_index = torch.argmax((~mask_original[i]).to(torch.float32)) * 320 + (
                    conditioning_voice_encoded.size(1) * 320
                )  # assuming encodec audio
                zero_masked_wave = wave[:, : int(conditioning_voice_encoded.size(1) * 320)]
                # truncate the wave following the pad mask
                wave = wave[:, int(conditioning_voice_encoded.size(1) * 320) : int(first_false_index)]
                try:
                    writer.add_audio(
                        f"One_mask_infilled_{name}/sample_{i}",
                        wave.detach().cpu().view(-1).unsqueeze(-1),
                        steps,
                        sample_rate=24_000,
                    )
                except Exception as e:
                    print("Conditional error", e)
                    pass

                if name == "1x":
                    try:
                        writer.add_audio(
                            f"Zero_mask_inilled/sample_{i}",
                            zero_masked_wave.detach().cpu().view(-1).unsqueeze(-1),
                            steps,
                            sample_rate=24_000,
                        )
                        writer.add_audio(
                            f"Conditioning_voice/sample_{i}",
                            conditioning_voice[i].detach().cpu().view(-1).unsqueeze(-1),
                            steps,
                            sample_rate=24_000,
                        )
                    except Exception as e:
                        print("Original error", e)
                        pass

                    # upload text
                    file_wav = dataset.data[idx_conditional[i]]["audio"]
                    with open(file_wav.replace(".wav", ".original.txt"), "r") as f:
                        text = f.read()
                    writer.add_text(f"Infilled/sample_{i}", text, steps)
