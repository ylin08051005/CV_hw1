# HW1:Group-housed Swine Object Detection

Use YOLOv8-based object detection system for group-housed swine with TTA (Test Time Augmentation) and model ensemble.
 
## Project Structure

```

hw1_111205039.zip
 |-- hw1_111205039
   |-------- report_111205039.pdf
   |-------- code_111205039.zip                     #Includes all complete source code, split datasets, training logs, and weight files.
      |-------- src/
                ├── train.py                           # Basic training script
                ├── train_improve.py                   # Improved training version()
                └── tta_ensemble.py                    # Test Time Augmentation (TTA) and model ensemble for inference
      |-------- readme.md # Instructions for environment setup and execution
      └-------- requirements.txt # List of required packages

```

## Environment Setup

### 1. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n pig_detection python=3.9
conda activate pig_detection

# Or using virtualenv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```


---

## 📁 Dataset Preparation (Important)

Please **place the dataset inside the `src/` folder** before running the training or prediction scripts.
After downloading the provided dataset, **rename and organize** the files into the following structure:

```
src/
├── gt.txt                  # Ground truth annotations (format: img_id,x,y,width,height)
├── img/                    # Training images
│   ├── 00000001.jpg
│   ├── 00000002.jpg
│   └── ...
└── test_images/            # Test images for prediction
    ├── 00000001.jpg
    ├── 00000002.jpg
    └── ...
```

### ⚠️ Notes

* The **`gt.txt`**, **`img/`**, and **`test_images/`** folders **must** be placed under `src/`.
* Make sure that the file names (`00000001.jpg`, etc.) match the annotation entries in `gt.txt`.
* Do **not** include the dataset in your final submission ZIP — instead, describe where to download it (e.g., Google Drive or Kaggle link) in the **README**.


---

是否要我幫你整合成完整的 `README.md` 範本（包含環境設定、訓練與推論指令）？這樣你可以直接交。


### Dataset Structure of code_111205039

```
code_111205039/
├── README.md                           # Instructions for environment setup and execution
├── requirements.txt                    # List of required packages
├── pig_detection/                       # Model 1 training output
│   ├── exp/
│   │   └── weights/
│   │       ├── best.pt                 # Best checkpoint of Model 1
│   │       └── last.pt                 # Last epoch checkpoint
│   ├── results.png                      # Training metrics plot
│   └── confusion_matrix.png             # Confusion matrix
├── pig_detection_improved/              # Model 2 training output
│   ├── exp/
│   │   └── weights/
│   │       └── best.pt                 # Best checkpoint of Model 2
│   ├── results.png
│   └── confusion_matrix.png
├── train.py                             # Basic training script (Model 1)
├── train_improve.py                      # Improved training script (Model 2)
├── tta_ensemble.py                       # TTA + Ensemble prediction script
├── img/                                  # Training images (if included)
├── gt.txt                                # Ground truth annotations
├── test_images/                          # Test images for prediction
├── submission.csv                        # Baseline submission
├── submission_improve.csv                # Improved model submission
├── submission_tta_ensemble.csv           # TTA + Ensemble submission
├── yolov8n.pt                            # YOLOv8n pretrained weights
├── yolov8s.pt                            # YOLOv8s pretrained weights
└── yolo_dataset/                        # YOLO-formatted dataset (already split into train & val)
    ├── dataset.yaml
    ├── images/                          # Includes split datasets (train and val data)
    └── labels/                          # Includes split datasets (train and val data)

```

### Test Dataset

Place test images in `test_images/` folder with format:`00000001.jpg`

## Training

### Step 1: First Training (Model 1)

```bash
python train.py
```

- Model: YOLOv8s
- Epochs: 150
- Image size: 640×640
- Batch size: 16

**Output:** `pig_detection/exp/weights/best.pt`


### Step 2: Second Training (Model 2)

Run training again with same or different configuration:

```bash
python train_improved.py
```
**Output:** `pig_detection_improved/exp/weights/best.pt`


### Step 3: Prediction

Generate Final Submission with TTA + Ensemble

```bash
python tta_ensemble.py
```

Required weights:

pig_detection/exp/weights/best.pt (from Model 1)

pig_detection_improved/exp/weights/best.pt (from Model 2)

IoU threshold: 0.5

Confidence threshold: 0.01

**Output:** `submission_tta_ensemble.csv`


Note: TTA + Ensemble combines predictions from both trained models to improve accuracy. Make sure both weight files are present in their respective folders before running this script.

## Model Architecture

- **Base Model:** YOLOv8s
- **TTA Strategies:** 
  - Original image
  - Horizontal flip
  - Multi-scale (0.9×, 1.1×)
- **Ensemble:** Weighted Boxes Fusion with 2 models


## Output Files

### Training Results

After training, you'll find:

**pig_detection/** (Model 1)
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Last epoch checkpoint
- `results.png` - Training metrics plot
- `confusion_matrix.png` - Confusion matrix
- `train_batch*.jpg` - Training batch visualizations

**pig_detection_improved/** (Model 2)
- Same structure as above

### Submission Files

- `submission_test.csv` - Predictions from baseline model
- `submission_improved.csv` - Predictions from improved model
- `submission_tta_ensemble.csv` - Predictions from TTA + ensemble (best)

### Submission Format

Each CSV file contains:
```
Image_ID,PredictionString
1,conf1 x1 y1 w1 h1 0 conf2 x2 y2 w2 h2 0 ...
2,conf1 x1 y1 w1 h1 0
3,
...
```

Where:
- `conf`: Confidence score (0-1)
- `x, y`: Top-left corner coordinates
- `w, h`: Box width and height
- `0`: Class ID (always 0 for pig)

Empty string means no detection for that image.

```

## Results(private score)

- First training(model 1): mAP ≈ 0.30790
- Second training(model 2): mAP ≈ 0.33176
- TTA + Ensemble: mAP ≈ 0.33936

