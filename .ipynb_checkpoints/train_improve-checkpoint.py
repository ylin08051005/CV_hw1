"""
Improved YOLOv8 Training Script
Goal: Improve mAP from 0.30 to 0.40+
"""

from ultralytics import YOLO
from pathlib import Path

def train_improved_model(
    yaml_path,
    model_size='s',
    epochs=200,
    img_size=640,
    batch_size=16
):
    """
    Improved training function
    
    Parameters:
        model_size: Model size
            'n' = nano (fastest, lowest accuracy)
            's' = small (recommended, balanced)
            'm' = medium (slower, higher accuracy)
            'l' = large (very slow, very high accuracy)
            'x' = xlarge (slowest, highest accuracy)
    """
    print("=" * 60)
    print(f"Starting training YOLOv8{model_size.upper()} model")
    print("=" * 60)
    
    # Load model
    model_name = f'yolov8{model_size}.pt'
    model = YOLO(model_name)
    print(f"Model loaded: {model_name}")
    
    # Training parameters (optimized)
    results = model.train(
        # Basic parameters
        data=str(yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device='cpu',  # Change to 0 to use GPU
        
        # Optimizer parameters
        optimizer='AdamW',      # Use AdamW (more stable than SGD)
        lr0=0.001,              # Initial learning rate
        lrf=0.01,               # Final learning rate (relative to lr0)
        momentum=0.937,         # SGD momentum
        weight_decay=0.0005,    # Weight decay
        warmup_epochs=3.0,      # Learning rate warmup epochs
        warmup_momentum=0.8,    # Warmup momentum
        
        # Data augmentation parameters
        hsv_h=0.015,           # Hue augmentation (0-1)
        hsv_s=0.7,             # Saturation augmentation (0-1)
        hsv_v=0.4,             # Value augmentation (0-1)
        degrees=10.0,          # Rotation angle (±deg)
        translate=0.1,         # Translation (fraction)
        scale=0.5,             # Scaling (gain)
        shear=0.0,             # Shear angle (deg)
        perspective=0.0,       # Perspective transform (0-0.001)
        flipud=0.0,            # Vertical flip probability
        fliplr=0.5,            # Horizontal flip probability
        mosaic=1.0,            # Mosaic augmentation probability
        mixup=0.1,             # MixUp augmentation probability
        copy_paste=0.1,        # Copy-Paste augmentation probability
        
        # Other parameters
        patience=50,           # Early stopping patience
        save=True,
        plots=True,
        exist_ok=True,
        project='pig_detection_improved',
        name='exp',
        verbose=True,
        
        # Advanced parameters
        cos_lr=True,           # Use Cosine learning rate scheduler
        close_mosaic=10,       # Close mosaic in last N epochs
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model weights: pig_detection_improved/exp/weights/best.pt")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Improved YOLOv8 Training Script")
    print("Goal: mAP 0.30 to 0.40+")
    print("=" * 60)
    
    # Configuration parameters
    YAML_PATH = "yolo_dataset/dataset.yaml"
    
    # Configuration 1: Quick improvement
    print("\nConfiguration: Quick Improvement")
    print("  Model: YOLOv8s")
    print("  Epochs: 150")
    print("  Image size: 640")
    print("  Expected improvement: +0.05~0.10 mAP")
    print("  Training time: ~2-3 hours (CPU)")
    print("=" * 60)
    
    # Train model with configuration 1
    model = train_improved_model(
        yaml_path=YAML_PATH,
        model_size='s',      # YOLOv8s
        epochs=150,
        img_size=640,
        batch_size=16        # Change to 8 if memory insufficient
    )
    
    # Generate new submission file
    print("\n" + "=" * 60)
    print("Generating new submission file...")
    print("=" * 60)
    
    import csv
    
    model_path = 'pig_detection_improved/exp/weights/best.pt'
    model = YOLO(model_path)
    
    with open('submission_improved.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_ID', 'PredictionString'])
        
        detected = 0
        
        for img_id in range(1, 1271):
            img_path = Path('test_images') / f"{img_id:08d}.jpg"
            
            if not img_path.exists():
                writer.writerow([img_id, ''])
                continue
            
            results = model.predict(img_path, conf=0.01, verbose=False)
            
            pred_string = []
            if len(results[0].boxes) > 0:
                detected += 1
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    x, y = xyxy[0], xyxy[1]
                    w, h = xyxy[2] - x, xyxy[3] - y
                    pred_string.append(f"{conf:.6f} {x:.2f} {y:.2f} {w:.2f} {h:.2f} 0")
            
            writer.writerow([img_id, ' '.join(pred_string)])
            
            if img_id % 200 == 0:
                print(f"  Progress: {img_id}/1270, Detected: {detected}")
    
    print(f"\nCompleted! Detected: {detected}/1270")
    print(f"Submission file: submission_improved.csv")