"""
EggplantGuard - YOLO Classification Model Training Script
=========================================================
Trains an Ultralytics YOLO classification model to classify eggplant leaves and fruits:
  1. Healthy Leaf
  2. Leaf Miner Damage
  3. Fresh/Market-Ready Fruit
  4. Damaged/Borer-Infested Fruit

Supports both:
  - Standard split structure: dataset/train/<class>, dataset/val/<class>
  - Flat folder structure: dataset/<class> (automatically splits 80% train / 20% val)
"""

import os
import sys
import shutil
import random
import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = [
    "Healthy Leaf",
    "Leaf Miner Damage",
    "Fresh-Market-Ready Fruit",
    "Damaged-Borer-Infested Fruit"
]

# Aliases to accommodate common folder naming variations
CLASS_ALIASES = {
    "healthy leaf": "Healthy Leaf",
    "healthy": "Healthy Leaf",
    "healthy_leaf": "Healthy Leaf",
    "leaf miner damage": "Leaf Miner Damage",
    "leaf miner": "Leaf Miner Damage",
    "leaf_miner": "Leaf Miner Damage",
    "fresh/market-ready fruit": "Fresh-Market-Ready Fruit",
    "fresh-market-ready fruit": "Fresh-Market-Ready Fruit",
    "fresh fruit": "Fresh-Market-Ready Fruit",
    "fresh": "Fresh-Market-Ready Fruit",
    "fresh_fruit": "Fresh-Market-Ready Fruit",
    "damaged/borer-infested fruit": "Damaged-Borer-Infested Fruit",
    "damaged-borer-infested fruit": "Damaged-Borer-Infested Fruit",
    "borer damage": "Damaged-Borer-Infested Fruit",
    "borer infested": "Damaged-Borer-Infested Fruit",
    "damaged fruit": "Damaged-Borer-Infested Fruit",
    "damaged_fruit": "Damaged-Borer-Infested Fruit"
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def check_dependencies():
    """Verify ultralytics and torch are installed."""
    try:
        import ultralytics
        import torch
        print(f"[INFO] Ultralytics version: {ultralytics.__version__}")
        print(f"[INFO] PyTorch version: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
        return True
    except ImportError as e:
        print(f"[ERROR] Required dependency missing: {e}")
        print("Please install requirements first:")
        print("  pip install -r requirements.txt")
        return False

def count_images_in_dir(directory: Path):
    """Count valid image files in a folder."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS)

def prepare_dataset():
    """
    Checks if dataset is in YOLO format (dataset/train and dataset/val).
    If dataset is flat (dataset/<class>), automatically splits images into train (80%) and val (20%).
    """
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    
    # Check if train/val already exist with images
    train_has_images = any(count_images_in_dir(d) > 0 for d in train_dir.iterdir() if d.is_dir()) if train_dir.exists() else False
    
    if train_has_images:
        print("[INFO] Found existing train/val dataset structure.")
        return DATASET_DIR

    # Check for flat class directories under dataset/
    subdirs = [d for d in DATASET_DIR.iterdir() if d.is_dir() and d.name not in {"train", "val", "test"}]
    
    if not subdirs:
        print("[ERROR] No dataset images found in 'dataset/'!")
        print("\nPlease structure your images in either:")
        print("  dataset/train/<class_name>/ and dataset/val/<class_name>/")
        print("OR:")
        print("  dataset/<class_name>/ (with images placed directly inside)")
        print("\nTarget classes:")
        for c in TARGET_CLASSES:
            print(f"  - {c}")
        return None

    print("[INFO] Flat dataset detected. Automatically splitting into 80% train / 20% val...")
    
    for subdir in subdirs:
        norm_name = CLASS_ALIASES.get(subdir.name.lower(), subdir.name)
        images = [f for f in subdir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
        
        if not images:
            continue
            
        random.seed(42)
        random.shuffle(images)
        
        split_idx = int(len(images) * 0.8)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        dest_train = train_dir / norm_name
        dest_val = val_dir / norm_name
        dest_train.mkdir(parents=True, exist_ok=True)
        dest_val.mkdir(parents=True, exist_ok=True)
        
        for img in train_imgs:
            shutil.copy2(img, dest_train / img.name)
        for img in val_imgs:
            shutil.copy2(img, dest_val / img.name)
            
        print(f"  Processed {norm_name}: {len(train_imgs)} train, {len(val_imgs)} val images.")

    return DATASET_DIR

def verify_dataset(dataset_path: Path):
    """Verify that dataset has at least some training and validation images."""
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        return False, "Missing 'train' or 'val' folder in dataset."
        
    train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
    if len(train_classes) < 2:
        return False, f"At least 2 classes required for classification, found {len(train_classes)} in train/."
        
    total_train = sum(count_images_in_dir(d) for d in train_classes)
    total_val = sum(count_images_in_dir(d) for d in val_dir.iterdir() if d.is_dir())
    
    print(f"[INFO] Dataset verification:")
    print(f"  Classes found in train: {[d.name for d in train_classes]}")
    print(f"  Total train images: {total_train}")
    print(f"  Total val images:   {total_val}")
    
    if total_train == 0:
        return False, "No images found in dataset/train subfolders. Please add real eggplant images before training."
        
    return True, "Dataset ready."

def train_model(epochs=25, imgsz=224, batch_size=16, base_model="yolov8n-cls.pt"):
    """Train YOLO classification model."""
    from ultralytics import YOLO
    
    print(f"\n{'='*60}")
    print("STARTING EGGPLANTGUARD YOLO TRAINING")
    print(f"Pretrained base: {base_model}")
    print(f"Epochs: {epochs} | Image size: {imgsz}x{imgsz} | Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    model = YOLO(base_model)
    
    results = model.train(
        data=str(DATASET_DIR),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project="runs/classify",
        name="eggplantguard",
        exist_ok=True,
        workers=2,
        verbose=True
    )
    
    # Locate best weights
    best_weights = BASE_DIR / "runs" / "classify" / "eggplantguard" / "weights" / "best.pt"
    if not best_weights.exists():
        # Fallback search
        found = list((BASE_DIR / "runs").glob("**/weights/best.pt"))
        if found:
            best_weights = found[-1]
            
    if best_weights.exists():
        dest_weights = MODEL_DIR / "best.pt"
        shutil.copy2(best_weights, dest_weights)
        print(f"\n[SUCCESS] Best model weights saved to: {dest_weights}")
        
        # Save model metadata
        trained_model = YOLO(dest_weights)
        metadata = {
            "model_type": "Ultralytics YOLO Classification",
            "base_model": base_model,
            "imgsz": imgsz,
            "classes": trained_model.names,
            "num_classes": len(trained_model.names),
            "weights_path": "model/best.pt"
        }
        with open(MODEL_DIR / "model_info.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"[INFO] Model metadata saved to: {MODEL_DIR / 'model_info.json'}")
        print("\nTraining completed successfully! You can now run 'python app.py'.")
    else:
        print("[WARNING] Could not automatically locate best.pt. Please check the 'runs/classify' directory.")

def main():
    print("=== EggplantGuard YOLO Model Trainer ===")
    if not check_dependencies():
        sys.exit(1)
        
    dataset_path = prepare_dataset()
    if not dataset_path:
        sys.exit(1)
        
    ok, message = verify_dataset(dataset_path)
    if not ok:
        print(f"\n[ERROR] {message}")
        print("\nHow to prepare your images:")
        print("1. Collect genuine photos of eggplant leaves and fruits.")
        print("2. Put healthy leaves in: 'dataset/train/Healthy Leaf'")
        print("3. Put leaf miner leaves in: 'dataset/train/Leaf Miner Damage'")
        print("4. Put fresh fruits in: 'dataset/train/Fresh-Market-Ready Fruit'")
        print("5. Put borer-damaged fruits in: 'dataset/train/Damaged-Borer-Infested Fruit'")
        print("6. Place a 20% validation split in the corresponding 'dataset/val/' folders.")
        print("7. Re-run 'python train_yolo.py'.")
        sys.exit(1)
        
    train_model()

if __name__ == "__main__":
    main()
