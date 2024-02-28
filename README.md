# beats
```
git clone https://github.com/hahmadraza/beats.git
cd beats
pip install -r requirements.txt
```

For Linux/MacOs
```
pyinstaller --onefile --clean --add-data "class_labels_indices.csv:." --add-data "weights:weights" BEATs_inference.py
```

For Windows
```
pyinstaller --onefile --clean --add-data "class_labels_indices.csv;." --add-data "weights;weights" BEATs_inference.py
```

Generated executeable will be locaterd in dist folder
