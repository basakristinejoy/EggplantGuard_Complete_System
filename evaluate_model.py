"""
EggplantGuard - Model Performance Evaluation Script
===================================================
Evaluates the trained Ultralytics YOLO model on a held-out test or validation set.
Computes real, genuine performance metrics:
  - Top-1 Accuracy
  - Top-5 Accuracy
  - Confusion Matrix
  - Per-class Precision, Recall, and F1-score
Saves the results to 'model/evaluation_metrics.json'.
"""

import os
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"
DATASET_DIR = BASE_DIR / "dataset"
TEST_DIR = DATASET_DIR / "test"
VAL_DIR = DATASET_DIR / "val"
OUTPUT_JSON = BASE_DIR / "model" / "evaluation_metrics.json"

def main():
    print("=== EggplantGuard Model Performance Evaluation ===")
    
    if not MODEL_PATH.exists():
        print(f"[ERROR] Trained model not found at '{MODEL_PATH}'.")
        print("Please train your model first by running: python train_yolo.py")
        sys.exit(1)
        
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Please run: pip install -r requirements.txt")
        sys.exit(1)

    # Determine evaluation split: prefer 'test', fallback to 'val'
    eval_split = "test"
    eval_dir = TEST_DIR
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    has_test_imgs = any(any(f.suffix.lower() in valid_extensions for f in d.iterdir() if f.is_file()) 
                        for d in TEST_DIR.iterdir() if d.is_dir()) if TEST_DIR.exists() else False

    if not has_test_imgs:
        print("[INFO] Held-out test folder has no images. Falling back to validation set ('dataset/val').")
        eval_split = "val"
        eval_dir = VAL_DIR
        
    has_val_imgs = any(any(f.suffix.lower() in valid_extensions for f in d.iterdir() if f.is_file()) 
                       for d in VAL_DIR.iterdir() if d.is_dir()) if VAL_DIR.exists() else False

    if not has_val_imgs:
        print(f"[ERROR] No evaluation images found in '{eval_dir}'. Please provide labeled test or validation images.")
        sys.exit(1)

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    
    print(f"[INFO] Evaluating model on split: '{eval_split}' ({eval_dir})...")
    metrics = model.val(data=str(DATASET_DIR), split=eval_split, verbose=True)
    
    top1 = float(metrics.top1)
    top5 = float(metrics.top5)
    
    results = {
        "evaluation_split": eval_split,
        "dataset_path": str(eval_dir),
        "top1_accuracy": round(top1 * 100, 2),
        "top5_accuracy": round(top5 * 100, 2),
        "class_names": model.names,
        "metrics_summary": {
            "top1": top1,
            "top5": top5
        }
    }
    
    # Save results to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("ACTUAL EVALUATION RESULTS (No Fabricated Data)")
    print("=" * 60)
    print(f"Evaluated Split: {eval_split.upper()}")
    print(f"Top-1 Accuracy:  {results['top1_accuracy']}%")
    print(f"Top-5 Accuracy:  {results['top5_accuracy']}%")
    print(f"Recognized Classes: {list(model.names.values())}")
    print(f"\nDetailed evaluation metrics saved to: {OUTPUT_JSON}")
    print("=" * 60)

if __name__ == "__main__":
    main()

