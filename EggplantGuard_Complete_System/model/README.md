# EggplantGuard Model Weights & Metadata

This directory stores the trained Ultralytics YOLO model files.

## Files Generated During Training:
- **`best.pt`**: The fine-tuned Ultralytics YOLO classification weights exported by `train_yolo.py`.
- **`model_info.json`**: Model architecture metadata, class mapping, and training timestamp.
- **`evaluation_metrics.json`**: Real accuracy, precision, recall, and top-1/top-5 metrics produced by `evaluate_model.py`.

When you run `python train_yolo.py`, `best.pt` will be automatically placed here, and `app.py` will load it immediately.
