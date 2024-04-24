import torch
import argparse
from config import config
from tokenizer import Tokenizer
from trainer import VoiceBoxTrainer
from hubert_kmeans import HubertWithKmeans
from data import AudioDataset, get_dataloader
from voicebox_pytorch import VoiceBox, EncodecVoco, ConditionalFlowMatcherWrapper, TextToSemantic

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="hubert/hubert_base_ls960.pt")
parser.add_argument("--kmeans_path", type=str, default="hubert/hubert_base_ls960_L9_km2000_expresso.bin")
parser.add_argument("--model_path", type=str)
parser.add_argument(
    "--audio_path", type=str, default="/net/tscratch/people/plgpiermelucci/data/LibriTTS/train-clean-360"
)

if __name__ == "__main__":
    """Train example without text_to_semantic SpearTTS model."""
    args = parser.parse_args()
    # phonem tokenizer
    tokenizer = Tokenizer(config)
    # dataset
    dataset = AudioDataset(folder=args.audio_path, json_pathlist="valid_audio_files.json", tokenizer=tokenizer)

    # prepare cfm wrapper
    # wav2vec = HubertWithKmeans(checkpoint_path=args.checkpoint_path, kmeans_path=args.kmeans_path)
    model = VoiceBox(dim=512, audio_enc_dec=EncodecVoco(), num_cond_tokens=500, depth=2, dim_head=64, heads=16)
    cfm_wrapper = ConditionalFlowMatcherWrapper(voicebox=model)

    # Let's train!
    trainer = VoiceBoxTrainer(cfm_wrapper=cfm_wrapper, dataset=dataset, lr=1e-4, batch_size=64, num_train_steps=1e6)
    trainer.train()
