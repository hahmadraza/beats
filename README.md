# beats
## Setting up the environment
Create condaenvironment and activate it
```
conda create -n beats_env python=3.8 -y
conda activate beats_env
```
## Cloning github repo and installing dependencies
Clone the repository and install requirements
```
git clone https://github.com/hahmadraza/beats.git
cd beats
pip install -r requirements.txt
```
Downlod weights from [here](https://drive.google.com/file/d/1233NK6I3z9TEUJobSHqd2B4RNtWn0UHR/view?usp=drivesdk) and put them inside weights folder in root directory 
```
|--beats
|  |--weights
      |--BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
   |--BEATS.py
   |--BEATS_inference.py
   ...
```
## Creating binary with pyinstaller
For Linux/MacOs
```
pyinstaller --onefile --clean --add-data "class_labels_indices.csv:." BEATs_inference.py
```

For Windows
```
pyinstaller --onefile --clean --add-data "class_labels_indices.csv;." BEATs_inference.py
```

Generated executeable will be locaterd in dist folder
Run following command to get list of arguments to run the exe file.
```
BEATs_inference.exe --help
```
```
usage: BEATs_inference.exe [-h] --weights WEIGHTS --input_file INPUT_FILE [--conf_thres CONF_THRES]
                          [--event_len EVENT_LEN] [--ringbuffer_length RINGBUFFER_LENGTH]
                          [--chunk_length CHUNK_LENGTH]

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     (Required) type=str. Path to model weights file.
  --input_file INPUT_FILE
                        (Required) type=str. Path to input audio file.
  --conf_thres CONF_THRES
                        (Optional) type=float, default=0.25. Sets the minimum confidence threshold for
                        detections. Events detected with confidence below this threshold will be discarded.    
                        Accepted value range [0.0, 1.0].
  --event_len EVENT_LEN
                        (Optional) type=int, default=500. Sets the minimum event length in miliseconds.        
                        Events detected with length below this will be discarded. Accepted value range [1,     
                        audio_length_in_miliseconds].
  --ringbuffer_length RINGBUFFER_LENGTH
                        (Optional) type=int, default=64000. Number of frames in buffer. Accepted value range   
                        [1, total_frames_in_audio].
  --chunk_length CHUNK_LENGTH
                        (Optional) type=int, default=1024. Number of frames to add in buffer in one
                        iteration. Accepted value rane [1, ringbuffer_length].
```
