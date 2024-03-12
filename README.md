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
pyinstaller --onefile --clean --add-data "class_labels_indices.csv:." --add-data "weights:weights" BEATs_inference.py
```

For Windows
```
pyinstaller --onefile --clean --add-data "class_labels_indices.csv;." --add-data "weights;weights" BEATs_inference.py
```

Generated executeable will be locaterd in dist folder
