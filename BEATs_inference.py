import argparse
import torch
from BEATs import BEATs, BEATsConfig
import sys
from pathlib import Path
import os
import pandas as pd
import warnings
import numpy as np
import librosa


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

# ##############################################################################
# # RING BUFFER
# ##############################################################################
class RingBuffer():
    """
    A 1D ring buffer using numpy arrays, designed to efficiently handle
    real-time audio buffering. Modified from
    https://scimusing.wordpress.com/2013/10/25/ring-buffers-in-pythonnumpy/
    """
    def __init__(self, length, dtype=np.float32):
        """
        :param int length: Number of samples in this buffer
        """

        self._length = length
        self._buf = np.zeros(length, dtype=dtype)
        self._bufrange = np.arange(length)
        self._idx = 0  # the oldest location

    def update(self, arr):
        """
        Adds 1D array to ring buffer. Note that ``len(arr)`` must be anything
        smaller than ``self.length``, otherwise it will error.
        """
        len_arr = len(arr)
        assert len_arr < self._length, "RingBuffer too small for this update!"
        idxs = (self._idx + self._bufrange[:len_arr]) % self._length
        self._buf[idxs] = arr
        self._idx = idxs[-1] + 1  # this will be the new oldest location

    def read(self):
        """
        Returns a copy of the whole ring buffer, unwrapped in a way that the
        first element is the oldest, and
          the last is the newest.
        """
        idxs = (self._idx + self._bufrange) % self._length  # read from oldest
        result = self._buf[idxs]
        return result

# File Audio Input Stream Class
class FileAudioInputStream:
    def __init__(self, filename, ringbuffer_length=64000, chunk_length=1024):
        self.filename = filename
        self.rb_length = ringbuffer_length
        self.chunk_length = chunk_length
        self.audio_data, self.sample_rate = librosa.load(self.filename, sr=None, mono=True)
        self.rb = RingBuffer(self.rb_length, np.float32)
        self.start_sample = 0
        self.current_position = 0

    def read(self):
        return self.rb.read()

    def start(self):
        pass

    def stop(self):
        pass

    def terminate(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    def update_ring_buffer_with_file_data(self):
        chunk = self.audio_data[self.start_sample:self.start_sample + self.chunk_length]
        if self.start_sample >= len(self.audio_data):
            return True
        self.rb.update(chunk)
        self.start_sample+=self.chunk_length
        self.current_position+=self.chunk_length
        if self.current_position > len(self.audio_data):
            self.current_position = len(self.audio_data)

def parse_opt():
    """Parses command-line arguments for BEATs inference, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="", required=True, help="(Required) type=str. Path to model weights file.")
    parser.add_argument("--input_file", type=str, default="", required=True, help="(Required) type=str. Path to input audio file.")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="(Optional) type=float, default=0.25. Sets the minimum confidence threshold for detections. Events detected with confidence below this threshold will be discarded. Accepted value range [0.0, 1.0].")
    parser.add_argument("--event_len", type=int, default=500, help="(Optional) type=int, default=500. Sets the minimum event length in miliseconds. Events detected with length below this will be discarded. Accepted value range [1, audio_length_in_miliseconds].")
    parser.add_argument("--ringbuffer_length", type=int, default=64000, help="(Optional) type=int, default=64000. Number of frames in buffer. Accepted value range [1, total_frames_in_audio].")
    parser.add_argument("--chunk_length", type=int, default=1024, help="(Optional) type=int, default=1024. Number of frames to add in buffer in one iteration. Accepted value rane [1, ringbuffer_length].")

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

    # output_file_name = opt.input_file.split('/')[-1].split('.')[0]+'.txt'
    joint_output_filename = opt.input_file.split('/')[-1].split('.')[0]+'_joint.txt'
    # output_path = Path(opt.input_file).parents[0] / output_file_name
    joint_output_path = Path(opt.input_file).parents[0] / joint_output_filename
    # if os.path.exists(output_path):
    #     os.remove(output_path)
    
    if os.path.exists(joint_output_path):
        os.remove(joint_output_path)

    print(f'Generating output file at {joint_output_path}')
    # print(f'Generating joint output file at {joint_output_path}')
    result_data = {}
    
    with FileAudioInputStream(opt.input_file, ringbuffer_length=opt.ringbuffer_length, chunk_length=opt.chunk_length) as audio_input:
        audio_input.start()

        while True:
            # Update the ring buffer with file data
            flag = audio_input.update_ring_buffer_with_file_data()
            if flag:
                break
            # Process the audio data as needed
            audio_data = audio_input.read()
            audio_tensor = torch.from_numpy(audio_data)

            audio_tensor = audio_tensor.unsqueeze(0)
            padding_mask = torch.zeros(audio_tensor.shape).bool()

            probs = BEATs_model.extract_features(audio_tensor, padding_mask=padding_mask)[0]
            top5_label_prob = [item[0] for item in zip(*probs.topk(k=5))][0].tolist()
            top5_label_idx = [item[1] for item in zip(*probs.topk(k=5))][0].tolist()
            
            top5_label = [checkpoint['label_dict'][label_idx] for label_idx in top5_label_idx]
            top5_label = [labels_mapper[label] for label in top5_label]
            timestamp = (audio_input.current_position/audio_input.sample_rate)*1000.0
            timestamp = round(timestamp)
            for l, p in zip(top5_label,top5_label_prob):
                if p>=opt.conf_thres:
                    # line = f'{timestamp}|{l}|{np.round(p, 3)}\n'
                    # insert_line(output_path, line)
                    if l in result_data:
                        result_data[l]["timestamps"].append(timestamp)
                        result_data[l]["probabilities"].append(p)
                    else:
                        result_data[l] = {"timestamps": [timestamp], "probabilities": [p]}

            # if not audio_data.any():
            #     break
        window_size = round((opt.chunk_length/audio_input.sample_rate)*1000)
        final_data = []
        for key in result_data.keys():
            timestamps = result_data[key]["timestamps"]
            probabilities = result_data[key]["probabilities"]
            temp_timestamps = [int(t/window_size) for t in timestamps]
            consecutive_sequences = find_consecutive_sequences(temp_timestamps)
            for start_ind, end_ind in consecutive_sequences:
                start_timestamp = timestamps[start_ind]
                end_timestamp = timestamps[end_ind]
                if start_ind==end_ind:
                    avg_prob = probabilities[start_ind]
                else:
                    avg_prob = sum(probabilities[start_ind:end_ind])/(end_ind-start_ind)

                if (end_timestamp-start_timestamp)>=opt.event_len:
                    final_data.append([start_timestamp, end_timestamp, key, avg_prob])
        sorted_results = sorted(final_data, key=lambda x: x[0])
        for arr in sorted_results:
            line = f'{arr[0]}|{arr[1]}|{arr[2]}|{round(arr[3],3)}\n'
            insert_line(joint_output_path, line)
    