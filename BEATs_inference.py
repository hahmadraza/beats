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
    parser.add_argument("--input_file", type=str, help="input audio file", required=True)
    parser.add_argument("--conf_thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--event_len", type=int, default=500, help="Minimum event length in miliseconds")

    opt = parser.parse_args()
    return opt

def find_consecutive_sequences(arr):
    sequences = []
    start, end = None, None

    for i in range(len(arr)):
        if i == 0 or arr[i] != arr[i - 1] + 1:
            # Start of a new sequence
            start = i
        end = i

        if i == len(arr) - 1 or arr[i] != arr[i + 1] - 1:
            # End of the current sequence
            sequences.append((start, end))

    return sequences


if __name__ == "__main__":
    opt = parse_opt()
    checkpoint = torch.load(opt.weights)
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()

    labels_df = pd.read_csv(ROOT /"class_labels_indices.csv")
    labels_mapper = labels_df.set_index('mid')['display_name'].to_dict()

    output_file_name = opt.input_file.split('/')[-1].split('.')[0]+'.txt'
    output_path = Path(opt.input_file).parents[0] / output_file_name
    if os.path.exists(output_path):
        os.remove(output_path)
    print(f'Generating output file at {output_path}')

    waveform, sample_rate = torchaudio.load(opt.input_file)
    waveform = torch.mean(waveform, dim=0)

    # Specify chunk size and step
    window_size = 500 #in miliseconds

    chunk_size = int((window_size/1000)*sample_rate)

    step = chunk_size 

    # Calculate the number of chunks
    num_chunks = (len(waveform) - chunk_size) // step + 1

    # Create a list of chunks
    chunks = {int((i * step*1000)/sample_rate): waveform[i * step: i * step + chunk_size] for i in range(num_chunks)}
    result_data = {}

    for key, value in chunks.items():
        audio_input = value.unsqueeze(0)
        padding_mask = torch.zeros(audio_input.shape).bool()

        probs = BEATs_model.extract_features(audio_input, padding_mask=padding_mask)[0]
        top5_label_prob = [item[0] for item in zip(*probs.topk(k=5))][0].tolist()
        top5_label_idx = [item[1] for item in zip(*probs.topk(k=5))][0].tolist()
        
        top5_label = [checkpoint['label_dict'][label_idx] for label_idx in top5_label_idx]
        top5_label = [labels_mapper[label] for label in top5_label]

        for l, p in zip(top5_label,top5_label_prob):
            if p>=opt.conf_thres:
                if l in result_data:
                    result_data[l]["timestamps"].append(key)
                    result_data[l]["probabilities"].append(p)
                else:
                    result_data[l] = {"timestamps": [key], "probabilities": [p]}
        
    final_data = []
    for key in result_data.keys():
        timestamps = result_data[key]["timestamps"]
        probabilities = result_data[key]["probabilities"]
        timestamps = [int(t/window_size) for t in timestamps]
        consecutive_sequences = find_consecutive_sequences(timestamps)
        for start_ind, end_ind in consecutive_sequences:
            start_timestamp = timestamps[start_ind]*window_size
            end_timestamp = timestamps[end_ind]*window_size + window_size
            if start_ind==end_ind:
                avg_prob = probabilities[start_ind]
            else:
                avg_prob = sum(probabilities[start_ind:end_ind])/(end_ind-start_ind)

            if (end_timestamp-start_timestamp)>=opt.event_len:
                final_data.append([start_timestamp, end_timestamp, key, avg_prob])
    sorted_results = sorted(final_data, key=lambda x: x[0])
    for arr in sorted_results:
        line = f'{arr[0]}|{arr[1]}|{arr[2]}|{round(arr[3],3)}\n'
        insert_line(output_path, line)
    