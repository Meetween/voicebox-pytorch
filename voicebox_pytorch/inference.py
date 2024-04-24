import torch
import argparse
import datetime
import torchaudio
from config import config
from einops import rearrange
from data import AudioDataset
from tokenizer import Tokenizer
from voicebox_pytorch import VoiceBox, EncodecVoco, ConditionalFlowMatcherWrapper
import random

# parse arguments
parser = argparse.ArgumentParser()
# parser.add_argument("--checkpoint_path", type=str, default="hubert/hubert_base_ls960.pt")
parser.add_argument("--kmeans_path", type=str, default="hubert/hubert_base_ls960_L9_km2000_expresso.bin")
parser.add_argument("--model_path", type=str)
parser.add_argument(
    "--audio_path", type=str, default="/net/tscratch/people/plgpiermelucci/data/LibriTTS/train-clean-360"
)
parser.add_argument("--checkpoint_path", type=str, default="results/voicebox.76000.pt")

if __name__ == "__main__":
    """Train example without text_to_semantic SpearTTS model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    # phonem tokenizer
    tokenizer = Tokenizer(config)
    # dataset
    dataset = AudioDataset(
        folder=args.audio_path, json_pathlist="valid_audio_files.json", tokenizer=tokenizer, reuturn_text=True
    )
    data = random.choice(dataset)
    audio, phoneme_ids, text = data["wave"].to(device), data["phoneme_ids"].to(device), data["text"]
    print(text)
    torchaudio.save("original.wav", audio.unsqueeze(0).cpu(), 24000)

    # prepare cfm wrapper
    model = VoiceBox(dim=512, audio_enc_dec=EncodecVoco(), num_cond_tokens=500, depth=2, dim_head=64, heads=16)
    cfm_wrapper = ConditionalFlowMatcherWrapper(voicebox=model, cond_drop_prob=0.2)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    cfm_wrapper.load_state_dict(checkpoint["model"])
    cfm_wrapper = cfm_wrapper.to(device)

    num_steps = 32
    cond_scale = 1.05

    # unconditional generation
    start_date = datetime.datetime.now()
    output_wave = cfm_wrapper.sample(phoneme_ids=phoneme_ids.unsqueeze(0), steps=num_steps, cond_scale=cond_scale)
    elapsed_time = (datetime.datetime.now() - start_date).total_seconds()

    output_wave = rearrange(output_wave, "1 1 n -> 1 n")
    output_duration = float(output_wave.shape[1]) / 24000
    realtime_mult = output_duration / elapsed_time

    print(
        f"\nGenerated sample of duration {output_duration:0.2f}s in {elapsed_time}s ({realtime_mult:0.2f}x realtime)\n\n"
    )

    # save audio
    torchaudio.save("unconditional.wav", output_wave.cpu(), 24000)

    # conditional generation
    infill_data = random.choice(dataset)
    audio_infill, phoneme_ids_infill, text_infill = (
        infill_data["wave"].to(device),
        infill_data["phoneme_ids"].to(device),
        infill_data["text"],
    )
    print(text_infill)
    torchaudio.save("infill.wav", audio_infill.unsqueeze(0).cpu(), 24000)
    start_date = datetime.datetime.now()
    cond = torch.cat([audio_infill.unsqueeze(0), audio.unsqueeze(0)], dim=-1).to(device)
    cond_mask = torch.cat(
        [
            torch.zeros_like(phoneme_ids_infill.unsqueeze(0)).to(torch.bool),
            torch.ones_like(phoneme_ids.unsqueeze(0)).to(torch.bool),
        ],
        dim=-1,
    ).to(device)
    output_wave = cfm_wrapper.sample(
        cond=cond,
        cond_mask=cond_mask,
        phoneme_ids=torch.cat([phoneme_ids_infill.unsqueeze(0), phoneme_ids.unsqueeze(0)], dim=-1),
        steps=num_steps,
        cond_scale=cond_scale,
    )
    elapsed_time = (datetime.datetime.now() - start_date).total_seconds()

    output_wave = rearrange(output_wave, "1 1 n -> 1 n")
    output_wave = output_wave[:, audio_infill.size(-1) :]
    output_duration = float(output_wave.shape[1]) / 24000
    realtime_mult = output_duration / elapsed_time

    print(
        f"\nGenerated sample of duration {output_duration:0.2f}s in {elapsed_time}s ({realtime_mult:0.2f}x realtime)\n\n"
    )

    # save audio
    torchaudio.save("conditional.wav", output_wave.cpu(), 24000)
