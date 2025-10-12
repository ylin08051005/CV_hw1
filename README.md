# HW1:Group-housed Swine Object Detection

Use YOLOv8-based object detection system for group-housed swine with TTA (Test Time Augmentation) and model ensemble.
 
## Project Structure

```

hw1_111205039.zip (e.g., hw1_R12345678.zip)
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

## Dataset Preparation

Ensure your dataset is organized as follows:

```
.
├── gt.txt                 # Format: img_id,x,y,width,height (one box per line)
├── img/                   # Training images
│   ├── 00000001.jpg
│   ├── 00000002.jpg
│   └── ...
└── test_images/           # Test images for prediction
    ├── 00000001.jpg
    ├── 00000002.jpg
    └── ...
```

### Training Dataset Structure

```
hw1_111205039.zip
└── hw1_111205039/
    ├── report_111205039.pdf
    ├── code_111205039.zip
    └── src/
        ├── train.py                 # Basic training script
        ├── train_improve.py         # Improved training version
        └── tta_ensemble.py          # Test Time Augmentation (TTA) and model ensemble for inference
    ├── readme.md                # Instructions for environment setup and execution
    └── requirements.txt         # List of required packages

    
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

