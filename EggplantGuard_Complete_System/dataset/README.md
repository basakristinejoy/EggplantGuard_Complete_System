# EggplantGuard Dataset Organization Guide

This directory holds your labeled training, validation, and testing images for **EggplantGuard**.

## 1. Directory Structure

```text
dataset/
├── train/
│   ├── Healthy Leaf/
│   ├── Leaf Miner Damage/
│   ├── Fresh-Market-Ready Fruit/
│   └── Damaged-Borer-Infested Fruit/
├── val/
│   ├── Healthy Leaf/
│   ├── Leaf Miner Damage/
│   ├── Fresh-Market-Ready Fruit/
│   └── Damaged-Borer-Infested Fruit/
└── test/
    ├── Healthy Leaf/
    ├── Leaf Miner Damage/
    ├── Fresh-Market-Ready Fruit/
    └── Damaged-Borer-Infested Fruit/
```

> **Convenience Note:** If you place images in flat folders directly under `dataset/` (e.g. `dataset/Healthy Leaf/`), `python train_yolo.py` will automatically split them into an 80% train / 20% validation split for you.

---

## 2. Image Collection Guidelines

### Target Categories:

1. **Healthy Leaf**:
   - Collect photos of clean, green eggplant leaves.
   - Include both upper (adaxial) and lower (abaxial) surfaces.
   - Varied lighting: morning sun, overcast, diffuse indoor light.
   - Avoid leaves that have dust or water splashes that mimic lesions.

2. **Leaf Miner Damage**:
   - Collect leaves showing characteristic white/grey winding serpentine mines (*Liriomyza* species).
   - Capture early mines (single thin trails) and advanced mines (dense intersecting trails).
   - Ensure the trails are in sharp focus.

3. **Fresh/Market-Ready Fruit**:
   - Mature, glossy purple (or green/white depending on cultivar) fruits.
   - Calyx should be intact and green.
   - No signs of bruising, mechanical tears, or insect holes.

4. **Damaged/Borer-Infested Fruit**:
   - Fruits damaged by Eggplant Fruit and Shoot Borer (*Leucinodes orbonalis*).
   - Visible pinholes, larval entrance/exit wounds, frass (sawdust-like excrement) around the calyx or fruit body.
   - Internal browning and rot caused by borer feeding.

---

## 3. Best Practices for High Accuracy
- **Image Count**: Aim for at least 100–250 real images per category (more is better).
- **Format**: `.jpg`, `.jpeg`, `.png`, or `.webp`.
- **Resolution**: At least 640x640 pixels (the model will resize to 224x224 during training).
- **Background Diversity**: Take photos in actual farm fields, garden plots, and clean white/neutral sorting tables.
- **Independence**: Keep the images in `dataset/test/` separate from training and validation. Never train on test images!
