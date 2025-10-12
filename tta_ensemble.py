"""
TTA + Model Ensemble Complete Solution
Combines multiple models + Test Time Augmentation
Expected improvement: +0.05~0.08 mAP
"""

from ultralytics import YOLO
from pathlib import Path
import csv
from PIL import Image
import numpy as np

class TTAEnsemble:
    """TTA + Model Ensemble Class"""
    
    def __init__(self, model_paths, conf_threshold=0.01):
        """
        Initialize
        
        Parameters:
            model_paths: List of model paths
            conf_threshold: Confidence threshold
        """
        self.models = []
        self.model_metrics = []
        self.conf_threshold = conf_threshold
        
        print("=" * 60)
        print("Loading models...")
        print("=" * 60)
        
        for i, path in enumerate(model_paths, 1):
            if Path(path).exists():
                model = YOLO(path)
                self.models.append(model)
                
                # Try to load model metrics
                metrics = self._load_model_metrics(path)
                self.model_metrics.append(metrics)
                
                print(f"Model {i}: {path}")
                if metrics:
                    print(f"  - mAP50: {metrics.get('mAP50', 'N/A'):.5f}")
                    print(f"  - mAP50-95: {metrics.get('mAP50-95', 'N/A'):.5f}")
                    print(f"  - Precision: {metrics.get('precision', 'N/A'):.5f}")
                    print(f"  - Recall: {metrics.get('recall', 'N/A'):.5f}")
            else:
                print(f"Warning: Model {i} not found: {path}")
        
        print(f"Loaded {len(self.models)} models")
        print("=" * 60)
    
    def _load_model_metrics(self, model_path):
        """Load training metrics from model results"""
        try:
            model_dir = Path(model_path).parent.parent
            results_csv = model_dir / 'results.csv'
            
            if results_csv.exists():
                import pandas as pd
                df = pd.read_csv(results_csv)
                
                # Get last epoch metrics
                last_row = df.iloc[-1]
                
                metrics = {
                    'box_loss': last_row.get('train/box_loss', None),
                    'cls_loss': last_row.get('train/cls_loss', None),
                    'dfl_loss': last_row.get('train/dfl_loss', None),
                    'precision': last_row.get('metrics/precision(B)', None),
                    'recall': last_row.get('metrics/recall(B)', None),
                    'mAP50': last_row.get('metrics/mAP50(B)', None),
                    'mAP50-95': last_row.get('metrics/mAP50-95(B)', None),
                    'val_box_loss': last_row.get('val/box_loss', None),
                    'val_cls_loss': last_row.get('val/cls_loss', None),
                    'val_dfl_loss': last_row.get('val/dfl_loss', None),
                }
                
                return {k: v for k, v in metrics.items() if v is not None}
        except Exception as e:
            print(f"  Warning: Could not load metrics - {e}")
        
        return None
    
    def predict_with_tta(self, img_path):
        """
        Predict single image with TTA + multiple models
        
        TTA strategy:
        1. Original image
        2. Horizontal flip
        3. Multi-scale (0.9x, 1.1x)
        
        Returns:
            List of all predicted boxes
        """
        all_predictions = []
        
        # Read image
        img = Image.open(img_path)
        img_w, img_h = img.size
        
        # TTA for each model
        for model_idx, model in enumerate(self.models):
            # 1. Original image
            results = model.predict(
                img_path,
                conf=self.conf_threshold,
                iou=0.45,
                verbose=False,
                augment=True  # YOLO built-in augmentation
            )
            all_predictions.extend(self._extract_boxes(results[0], model_idx, 'original'))
            
            # 2. Horizontal flip
            img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            results_flip = model.predict(
                img_flipped,
                conf=self.conf_threshold,
                iou=0.45,
                verbose=False
            )
            
            # Convert flipped coordinates back to original image
            for box in results_flip[0].boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                
                # Flip x coordinates
                x1_flip = img_w - xyxy[2]
                x2_flip = img_w - xyxy[0]
                y1, y2 = xyxy[1], xyxy[3]
                
                all_predictions.append({
                    'conf': conf,
                    'x': x1_flip,
                    'y': y1,
                    'w': x2_flip - x1_flip,
                    'h': y2 - y1,
                    'model': model_idx,
                    'aug': 'flip'
                })
            
            # 3. Multi-scale
            for scale in [0.9, 1.1]:
                new_size = (int(img_w * scale), int(img_h * scale))
                img_scaled = img.resize(new_size, Image.BILINEAR)
                
                results_scale = model.predict(
                    img_scaled,
                    conf=self.conf_threshold,
                    iou=0.45,
                    verbose=False
                )
                
                # Convert scaled coordinates back to original image
                for box in results_scale[0].boxes:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    
                    all_predictions.append({
                        'conf': conf,
                        'x': xyxy[0] / scale,
                        'y': xyxy[1] / scale,
                        'w': (xyxy[2] - xyxy[0]) / scale,
                        'h': (xyxy[3] - xyxy[1]) / scale,
                        'model': model_idx,
                        'aug': f'scale{scale}'
                    })
        
        return all_predictions
    
    def _extract_boxes(self, result, model_idx, aug_type):
        """Extract boxes from prediction results"""
        predictions = []
        if len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                predictions.append({
                    'conf': conf,
                    'x': xyxy[0],
                    'y': xyxy[1],
                    'w': xyxy[2] - xyxy[0],
                    'h': xyxy[3] - xyxy[1],
                    'model': model_idx,
                    'aug': aug_type
                })
        return predictions
    
    def weighted_boxes_fusion(self, predictions, iou_threshold=0.5, conf_threshold=0.01):
        """
        Weighted Boxes Fusion (simplified version)
        
        Strategy:
        1. Group similar boxes
        2. Take weighted average for each group
        3. Average confidence scores
        """
        if len(predictions) == 0:
            return []
        
        # Filter low confidence boxes
        predictions = [p for p in predictions if p['conf'] >= conf_threshold]
        
        if len(predictions) == 0:
            return []
        
        # Sort by confidence score
        predictions.sort(key=lambda x: x['conf'], reverse=True)
        
        # Box fusion
        fused_boxes = []
        used = [False] * len(predictions)
        
        for i, pred in enumerate(predictions):
            if used[i]:
                continue
            
            # Find all boxes that overlap with current box
            cluster = [pred]
            used[i] = True
            
            for j in range(i + 1, len(predictions)):
                if used[j]:
                    continue
                
                iou = self._calculate_iou(pred, predictions[j])
                if iou > iou_threshold:
                    cluster.append(predictions[j])
                    used[j] = True
            
            # Fuse this group of boxes
            if len(cluster) >= 2:
                # Weighted average (weight = confidence score)
                total_conf = sum(b['conf'] for b in cluster)
                
                fused_box = {
                    'x': sum(b['x'] * b['conf'] for b in cluster) / total_conf,
                    'y': sum(b['y'] * b['conf'] for b in cluster) / total_conf,
                    'w': sum(b['w'] * b['conf'] for b in cluster) / total_conf,
                    'h': sum(b['h'] * b['conf'] for b in cluster) / total_conf,
                    'conf': total_conf / len(cluster),  # Average confidence score
                }
            else:
                # Single box used directly
                fused_box = cluster[0]
            
            fused_boxes.append(fused_box)
        
        # Sort by confidence score again
        fused_boxes.sort(key=lambda x: x['conf'], reverse=True)
        
        return fused_boxes
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1_1, y1_1 = box1['x'], box1['y']
        x2_1, y2_1 = x1_1 + box1['w'], y1_1 + box1['h']
        
        x1_2, y1_2 = box2['x'], box2['y']
        x2_2, y2_2 = x1_2 + box2['w'], y1_2 + box2['h']
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = box1['w'] * box1['h']
        area2 = box2['w'] * box2['h']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def print_ensemble_metrics(self):
        """Print ensemble metrics summary"""
        if not self.model_metrics or not any(self.model_metrics):
            return
        
        print("\n" + "=" * 60)
        print("ENSEMBLE METRICS SUMMARY")
        print("=" * 60)
        
        # Calculate average metrics
        valid_metrics = [m for m in self.model_metrics if m]
        
        if valid_metrics:
            avg_metrics = {}
            for key in valid_metrics[0].keys():
                values = [m[key] for m in valid_metrics if key in m]
                if values:
                    avg_metrics[key] = sum(values) / len(values)
            
            print("\nAverage Training Metrics:")
            if 'box_loss' in avg_metrics:
                print(f"  train/box_loss: {avg_metrics['box_loss']:.5f}")
            if 'cls_loss' in avg_metrics:
                print(f"  train/cls_loss: {avg_metrics['cls_loss']:.5f}")
            if 'dfl_loss' in avg_metrics:
                print(f"  train/dfl_loss: {avg_metrics['dfl_loss']:.5f}")
            
            print("\nAverage Performance Metrics:")
            if 'precision' in avg_metrics:
                print(f"  metrics/precision(B): {avg_metrics['precision']:.5f}")
            if 'recall' in avg_metrics:
                print(f"  metrics/recall(B): {avg_metrics['recall']:.5f}")
            if 'mAP50' in avg_metrics:
                print(f"  metrics/mAP50(B): {avg_metrics['mAP50']:.5f}")
            if 'mAP50-95' in avg_metrics:
                print(f"  metrics/mAP50-95(B): {avg_metrics['mAP50-95']:.5f}")
            
            print("\nAverage Validation Metrics:")
            if 'val_box_loss' in avg_metrics:
                print(f"  val/box_loss: {avg_metrics['val_box_loss']:.5f}")
            if 'val_cls_loss' in avg_metrics:
                print(f"  val/cls_loss: {avg_metrics['val_cls_loss']:.5f}")
            if 'val_dfl_loss' in avg_metrics:
                print(f"  val/dfl_loss: {avg_metrics['val_dfl_loss']:.5f}")
            
            print("=" * 60)


def generate_tta_ensemble_submission(
    model_paths,
    test_folder,
    output_csv='submission_tta_ensemble.csv',
    conf_threshold=0.01,
    iou_threshold=0.5
):
    """
    Generate TTA + ensemble submission file
    
    Parameters:
        model_paths: List of model paths
        test_folder: Test images folder
        output_csv: Output CSV file
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
    """
    print("=" * 60)
    print("TTA + Model Ensemble")
    print("Standard Configuration")
    print("=" * 60)
    
    # Initialize TTA ensemble
    ensemble = TTAEnsemble(model_paths, conf_threshold)
    
    # Print ensemble metrics
    ensemble.print_ensemble_metrics()
    
    # Calculate expected time
    num_models = len(ensemble.models)
    num_augs = 4  # original + flip + 2 scales
    total_predictions = 1270 * num_models * num_augs
    estimated_time = total_predictions * 0.5 / 60  # minutes
    
    print(f"\nEstimated time: {estimated_time:.1f} minutes")
    print(f"Predictions per image: {num_models * num_augs} times\n")
    print("=" * 60)
    
    detected_count = 0
    total_boxes = 0
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_ID', 'PredictionString'])
        
        for img_id in range(1, 1271):
            img_path = Path(test_folder) / f"{img_id:08d}.jpg"
            
            if not img_path.exists():
                writer.writerow([img_id, ''])
                continue
            
            # TTA + multi-model prediction
            all_predictions = ensemble.predict_with_tta(str(img_path))
            
            # Box fusion
            fused_boxes = ensemble.weighted_boxes_fusion(
                all_predictions,
                iou_threshold=iou_threshold,
                conf_threshold=conf_threshold
            )
            
            if len(fused_boxes) > 0:
                detected_count += 1
                total_boxes += len(fused_boxes)
                
                # Format output
                pred_string = []
                for box in fused_boxes:
                    pred_string.append(
                        f"{box['conf']:.6f} {box['x']:.2f} {box['y']:.2f} "
                        f"{box['w']:.2f} {box['h']:.2f} 0"
                    )
                
                writer.writerow([img_id, ' '.join(pred_string)])
            else:
                writer.writerow([img_id, ''])
            
            # Display progress
            if img_id % 50 == 0:
                progress = img_id / 1270 * 100
                avg = total_boxes / detected_count if detected_count > 0 else 0
                print(f"  Progress: {img_id}/1270 ({progress:.1f}%) | "
                      f"Detected: {detected_count} images | Average: {avg:.1f} objects/image")
    
    # Final statistics
    print("\n" + "=" * 60)
    print(f"TTA + Ensemble submission file generated!")
    print("=" * 60)
    print(f"Detected: {detected_count}/1270 images ({detected_count/1270*100:.1f}%)")
    print(f"Total boxes: {total_boxes}")
    if detected_count > 0:
        print(f"Average per image: {total_boxes/detected_count:.1f} boxes")
    print(f"Submission file: {output_csv}")
    print("=" * 60)
    
    return output_csv


if __name__ == "__main__":
    # Set model paths (modify according to your actual paths)
    MODEL_PATHS = [
        'pig_detection-Copy1/exp/weights/best.pt',
        'pig_detection_improved-Copy1/exp/weights/best.pt',
    ]
    
    TEST_FOLDER = 'test_images'
    
    print("\n" + "=" * 60)
    print("Configuration: Standard (Recommended)")
    print("  - IoU threshold: 0.5")
    print("  - Confidence threshold: 0.01")
    print("  - Expected effect: Best")
    print("=" * 60)
    
    output = generate_tta_ensemble_submission(
        model_paths=MODEL_PATHS,
        test_folder=TEST_FOLDER,
        output_csv='submission_tta_ensemble.csv',
        conf_threshold=0.01,
        iou_threshold=0.5
    )
    
    print("\n" + "=" * 60)
    print("Completed!")
    print("=" * 60)
    print(f"Submission file: {output}")
    print("\nSuggestions:")
    print("  1. Upload to Kaggle to test score")
    print("  2. If results are poor, try adjusting thresholds")
    print("=" * 60)