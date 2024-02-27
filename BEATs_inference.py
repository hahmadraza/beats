import argparse

import torch
import torchaudio
from BEATs import BEATs, BEATsConfig
import sys
from pathlib import Path
import os
import pandas as pd
import warnings

# Ignore the specific warning
warnings.filterwarnings("ignore", category=UserWarning)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# Function to insert a pipe-delimited line into the file
def insert_line(file_path, data_to_insert):
    with open(file_path, 'a+') as file:
        file.write(data_to_insert)

def parse_opt():
    """Parses command-line arguments for BEATs inference, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt", help="model path")
    parser.add_argument("--input_file", type=str, help="input audio file")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--event_len", type=int, default=500, help="Minimum event length in miliseconds")

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    labels_df = pd.read_csv("class_labels_indices.csv")
    labels_mapper = labels_df.set_index('mid')['display_name'].to_dict()
    output_file_name = opt.input_file.split('/')[-1].split('.')[0]+'.txt'
    output_path = Path(opt.input_file).parents[0] / output_file_name
    if os.path.exists(output_path):
        os.remove(output_path)
    print(f'Generating output file at {output_path}')
    checkpoint = torch.load(opt.weights)
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()
    waveform, sample_rate = torchaudio.load(opt.input_file)
    waveform = torch.mean(waveform, dim=0)

    # Specify chunk size and step
    window_size = opt.event_len #in miliseconds
    chunk_size = int((window_size/1000)*sample_rate)

    step = 100 # step size in miliseconds
    step = int((step/1000)*sample_rate)
    # step = chunk_size

    # Calculate the number of chunks
    num_chunks = (len(waveform) - chunk_size) // step + 1

    # Create a list of chunks
    chunks = {int((i * step*1000)/sample_rate): waveform[i * step: i * step + chunk_size] for i in range(num_chunks)}

    for key, value in chunks.items():
        audio_input = value.unsqueeze(0)
        padding_mask = torch.zeros(audio_input.shape).bool()

        probs = BEATs_model.extract_features(audio_input, padding_mask=padding_mask)[0]

        for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
            top5_label = [checkpoint['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
            top5_label = [labels_mapper[label] for label in top5_label]

            for l, p in zip(top5_label,top5_label_prob):
                if p>=opt.conf_thres:
                    line = f'{key}|{l}|{round(p.cpu().item(),2)}\n'
                    insert_line(output_path, line)
