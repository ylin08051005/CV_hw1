"""
Pig Detection YOLOv8 Complete Training Pipeline
Suitable for beginners
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import yaml
import csv

# =============== Step 1: Install Required Packages ===============
# Run in terminal:
# pip install ultralytics pillow pandas scikit-learn pyyaml

# =============== Step 2: Split Training and Validation Sets ===============

def split_dataset(gt_path, train_ratio=0.8, random_seed=42):
    """
    Split dataset into training and validation sets
    
    Args:
        gt_path: Path to gt.txt file
        train_ratio: Training set ratio (0.8 = 80% train, 20% validation)
        random_seed: Random seed (ensures consistent splits)
    
    Returns:
        train_ids: List of training image IDs
        val_ids: List of validation image IDs
        annotations: All annotation data
    """
    print("=" * 50)
    print("Step 1: Split Training and Validation Sets")
    print("=" * 50)
    
    # Read gt.txt
    print(f"Reading annotation file: {gt_path}")
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    
    # Organize annotation data
    annotations = {}
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) != 5:
            continue
        
        img_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append((x, y, w, h))
    
    total_images = len(annotations)
    total_boxes = sum(len(boxes) for boxes in annotations.values())
    
    print(f"\nDataset Statistics:")
    print(f"  Total images: {total_images}")
    print(f"  Total bounding boxes: {total_boxes}")
    print(f"  Average boxes per image: {total_boxes/total_images:.1f}")
    
    # Split into training and validation sets
    img_ids = list(annotations.keys())
    train_ids, val_ids = train_test_split(
        img_ids, 
        train_size=train_ratio, 
        random_state=random_seed
    )
    
    train_boxes = sum(len(annotations[img_id]) for img_id in train_ids)
    val_boxes = sum(len(annotations[img_id]) for img_id in val_ids)
    
    print(f"\nSplit Results:")
    print(f"  Training set: {len(train_ids)} images, {train_boxes} boxes")
    print(f"  Validation set: {len(val_ids)} images, {val_boxes} boxes")
    print(f"  Ratio: {train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%")
    print("=" * 50)
    
    return train_ids, val_ids, annotations


# =============== Step 3: Convert Data Format ===============

def convert_gt_to_yolo(img_folder, output_folder, train_ids, val_ids, annotations):
    """
    Convert gt.txt to YOLO format
    
    Args:
        img_folder: Image folder path
        output_folder: Output folder
        train_ids: List of training image IDs
        val_ids: List of validation image IDs
        annotations: All annotation data
    """
    print("\n" + "=" * 50)
    print("Step 2: Convert Data Format to YOLO Format")
    print("=" * 50)
    
    # Create folder structure
    train_img_dir = Path(output_folder) / "images" / "train"
    val_img_dir = Path(output_folder) / "images" / "val"
    train_label_dir = Path(output_folder) / "labels" / "train"
    val_label_dir = Path(output_folder) / "labels" / "val"
    
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    success_count = 0
    error_count = 0
    
    for split_name, img_ids_list, img_dir, label_dir in [
        ("Training set", train_ids, train_img_dir, train_label_dir),
        ("Validation set", val_ids, val_img_dir, val_label_dir)
    ]:
        print(f"\nProcessing {split_name}...")
        split_success = 0
        
        for img_id in img_ids_list:
            # Find image file (support multiple filename formats)
            img_path = None
            # Try different filename formats
            filename_formats = [
                f"{img_id:08d}.jpg",       # 00000001.jpg (8 digits with leading zeros) ← 新增
                f"{img_id:08d}.JPG",       # 00000001.JPG ← 新增
                f"{img_id:08d}.png",       # 00000001.png ← 新增
                f"{img_id:08d}.PNG",       # 00000001.PNG ← 新增
                f"{img_id}.jpg",           # 1.jpg (no leading zeros)
                f"{img_id}.JPG",
                f"{img_id}.png",
                f"{img_id}.PNG",
            ]
            
            for filename in filename_formats:
                temp_path = Path(img_folder) / filename
                if temp_path.exists():
                    img_path = temp_path
                    break
            
            if img_path is None:
                print(f"  Warning: Image {img_id} not found")
                error_count += 1
                continue
            
            try:
                # Read image dimensions
                img = Image.open(img_path)
                img_w, img_h = img.size
                
                # Copy image
                shutil.copy(img_path, img_dir / f"{img_id}.jpg")
                
                # Convert annotations to YOLO format
                label_path = label_dir / f"{img_id}.txt"
                with open(label_path, 'w') as f:
                    for (x, y, w, h) in annotations[img_id]:
                        # Convert to center coordinates and normalize
                        x_center = (x + w/2) / img_w
                        y_center = (y + h/2) / img_h
                        width = w / img_w
                        height = h / img_h
                        
                        # Ensure coordinates are within [0, 1] range
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        # class_id = 0 (pig)
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                success_count += 1
                split_success += 1
                
            except Exception as e:
                print(f"  Error processing image {img_id}: {e}")
                error_count += 1
        
        print(f"  {split_name} completed: {split_success} images")
    
    print("\n" + "=" * 50)
    print(f"Conversion completed!")
    print(f"  Success: {success_count} images")
    print(f"  Failed: {error_count} images")
    print("=" * 50)
    
    return output_folder


# =============== Step 4: Create Configuration File ===============

def create_yaml_config(output_folder):
    """Create YOLO training configuration file"""
    
    print("\n" + "=" * 50)
    print("Step 3: Create YOLO Configuration File")
    print("=" * 50)
    
    config = {
        'path': str(Path(output_folder).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['pig']
    }
    
    yaml_path = Path(output_folder) / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration file created: {yaml_path}")
    print("=" * 50)
    return yaml_path


# =============== Step 5: Train Model ===============

def train_yolo_model(yaml_path, epochs=100, img_size=640, batch_size=16):
    """
    Train YOLO model
    
    Args:
        yaml_path: Path to dataset.yaml
        epochs: Number of training epochs
        img_size: Image size
        batch_size: Batch size (reduce to 8 or 4 if memory insufficient)
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 50)
    print("Step 4: Start Model Training")
    print("=" * 50)
    
    # Load pretrained model (options: yolov8n.pt, yolov8s.pt, yolov8m.pt)
    # n = nano (fastest but lower accuracy)
    # s = small (balanced)
    # m = medium (slower but higher accuracy)
    model = YOLO('yolov8n.pt')
    
    # Start training
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device="cpu",  # 0 = GPU, 'cpu' = CPU
        project='pig_detection',
        name='exp',
        patience=20,  # Early stopping
        save=True,
        plots=True,
        verbose=True
    )
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Model weights saved at: pig_detection/exp/weights/best.pt")
    print("=" * 50)
    
    return model


# =============== Step 6: Generate Submission File ===============

def generate_submission(model_path, test_folder, output_csv='submission.csv', conf_threshold=0.01):
    """
    Generate Kaggle submission file
    
    Args:
        model_path: Path to trained model weights
        test_folder: Test images folder
        output_csv: Output CSV filename
        conf_threshold: Confidence threshold (lower to detect more objects)
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("Step 5: Generate Kaggle Submission File")
    print("=" * 60)
    
    # Check if model path exists
    if not Path(model_path).exists():
        print(f"Error: Model weights file not found: {model_path}")
        print("Please confirm training is complete and model is saved")
        return
    
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Confidence threshold set to: {conf_threshold}")
    
    # Statistics
    total_images = 0
    found_images = 0
    images_with_detections = 0
    total_detections = 0
    missing_images = 0
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_ID', 'PredictionString'])
        
        for img_id in range(1, 1865):  # 1-1864
            total_images += 1
            
            # Try multiple filename formats
            img_path = None
            test_formats = [
                f"{img_id}.jpg",           # 1.jpg
                f"{img_id:08d}.jpg",       # 00000001.jpg
                f"{img_id}.png",           # 1.png
                f"{img_id:08d}.png",       # 00000001.png
                f"{img_id}.JPG",
                f"{img_id:08d}.JPG",
                f"{img_id}.jpeg",
                f"{img_id}.JPEG",
            ]
            
            for fmt in test_formats:
                temp_path = Path(test_folder) / fmt
                if temp_path.exists():
                    img_path = temp_path
                    found_images += 1
                    break
            
            if img_path is None:
                if missing_images < 5:  # Only show first 5 errors
                    print(f"  Warning: Test image {img_id} not found")
                missing_images += 1
                writer.writerow([img_id, ''])
                continue
            
            try:
                # Predict (lower confidence threshold for more detections)
                results = model.predict(img_path, conf=conf_threshold, verbose=False, iou=0.5)
                
                # Format prediction results
                pred_string = []
                if len(results[0].boxes) > 0:
                    images_with_detections += 1
                    total_detections += len(results[0].boxes)
                    
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        
                        # Calculate x, y, w, h (top-left coordinates + width/height)
                        x = xyxy[0]
                        y = xyxy[1]
                        w = xyxy[2] - xyxy[0]
                        h = xyxy[3] - xyxy[1]
                        
                        pred_string.append(f"{conf:.6f} {x:.2f} {y:.2f} {w:.2f} {h:.2f} 0")
                
                writer.writerow([img_id, ' '.join(pred_string)])
                
            except Exception as e:
                print(f"  Error processing image {img_id}: {e}")
                writer.writerow([img_id, ''])
            
            # Display progress and detection statistics
            if img_id % 200 == 0:
                progress = img_id / 1864 * 100
                avg_det = total_detections / images_with_detections if images_with_detections > 0 else 0
                print(f"  Progress: {img_id}/1864 ({progress:.1f}%) | Detected: {images_with_detections} images | Average: {avg_det:.1f} objects/image")
    
    # Final statistics
    print("\n" + "=" * 60)
    print(f"Submission file generated: {output_csv}")
    print("=" * 60)
    print(f"Detection Statistics:")
    print(f"  Total images: {total_images}")
    print(f"  Images found: {found_images} ({found_images/total_images*100:.1f}%)")
    print(f"  Missing images: {missing_images}")
    print(f"  Images with detections: {images_with_detections} ({images_with_detections/found_images*100:.1f}% of found)")
    print(f"  Total detection boxes: {total_detections}")
    if images_with_detections > 0:
        print(f"  Average boxes per image: {total_detections/images_with_detections:.1f}")
    print("=" * 60)
    
    # Warning messages
    if found_images < total_images * 0.9:
        print("\nWarning: Many test images not found!")
        print(f"  Only found {found_images}/{total_images} images")
        print("  Please check filename format in test_images folder")
    
    if images_with_detections < found_images * 0.5:
        print("\nWarning: Low detection rate!")
        print(f"  Only {images_with_detections}/{found_images} images have detections")
        print("\nPossible reasons:")
        print("  1. Test set differs significantly from training set")
        print("  2. Insufficient training (too few epochs)")
        print("  3. Confidence threshold still too high")
        print("\nSuggestions:")
        print("  - Check training results: pig_detection/exp/results.png")
        print("  - Try lower confidence threshold: conf_threshold=0.001")
        print("  - Use larger model: yolov8s.pt or yolov8m.pt")
        print("  - Increase training epochs: epochs=200")
    
    return output_csv


# =============== Main Program ===============

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════╗
    ║   Pig Detection YOLOv8 Complete Pipeline  ║
    ║   (Dataset split before processing)       ║
    ╚═══════════════════════════════════════════╝
    """)
    
    # ========== Configure Paths ==========
    GT_FILE = "gt.txt"              # Annotation file
    IMG_FOLDER = "img"              # Training images folder
    TEST_FOLDER = "test_images"     # Test images folder
    OUTPUT_FOLDER = "yolo_dataset"  # Output folder
    
    # ========== Step 1: Split Dataset ==========
    train_ids, val_ids, annotations = split_dataset(
        gt_path=GT_FILE,
        train_ratio=0.8,  # 80% train, 20% validation
        random_seed=42    # Fixed random seed
    )
    
    # ========== Step 2: Convert Data ==========
    convert_gt_to_yolo(
        img_folder=IMG_FOLDER,
        output_folder=OUTPUT_FOLDER,
        train_ids=train_ids,
        val_ids=val_ids,
        annotations=annotations
    )
    
    # ========== Step 3: Create Configuration File ==========
    yaml_path = create_yaml_config(OUTPUT_FOLDER)
    
    # ========== Step 4: Train Model ==========
    model = train_yolo_model(
        yaml_path=yaml_path,
        epochs=100,      # Training epochs
        img_size=640,    # Image size
        batch_size=16    # Batch size (reduce if memory insufficient)
    )
    
    # ========== Step 5: Generate Submission File ==========
    generate_submission(
        model_path='pig_detection/exp/weights/best.pt',
        test_folder=TEST_FOLDER,
        output_csv='submission_test.csv',
        conf_threshold=0.01  # Low threshold for more detections
    )
    
    print("\nAll completed! You can now upload submission.csv to Kaggle!")
    
    # ========== Output Final Statistics ==========
    print("\n" + "=" * 50)
    print("Final Statistics")
    print("=" * 50)
    print(f"Training images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")
    print(f"Model weights: pig_detection/exp/weights/best.pt")
    print(f"Submission file: submission.csv")
    print("=" * 50)