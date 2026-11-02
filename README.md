# EggplantGuard: Eggplant Fruit Quality and Leaf Disease Classification System

An end-to-end, full-stack AI web application engineered to diagnose eggplant (*Solanum melongena*) leaf diseases and evaluate harvested fruit quality. 

EggplantGuard utilizes **Ultralytics YOLO** for computer vision classification, **Python Flask** for the backend API, **SQLite** for audit trail persistence, and a custom **responsive agricultural-themed HTML5/CSS3/JavaScript interface**.

> **Note on AI Methodology:** This system uses **Ultralytics YOLO classification**, avoiding Google Teachable Machine or generic third-party cloud classifiers. All image models and weights are trained directly on your dataset.

---

## 📋 Table of Contents
1. [System Architecture](#system-architecture)
2. [Target Classification Categories](#target-classification-categories)
3. [File and Directory Structure](#file-and-directory-structure)
4. [Step-by-Step Installation & Execution Guide](#step-by-step-installation--execution-guide)
   - [Step 1: Install Python](#step-1-install-python)
   - [Step 2: Install Visual Studio Code](#step-2-install-visual-studio-code)
   - [Step 3: Extract the Project Archive](#step-3-extract-the-project-archive)
   - [Step 4: Open Project Folder in VS Code](#step-4-open-project-folder-in-vs-code)
   - [Step 5: Create Virtual Environment & Install Dependencies](#step-5-create-virtual-environment--install-dependencies)
   - [Step 6: Prepare the Eggplant Dataset](#step-6-prepare-the-eggplant-dataset)
   - [Step 7: Train the YOLO Classification Model](#step-7-train-the-yolo-classification-model)
   - [Step 8: Evaluate & Connect Model](#step-8-evaluate--connect-model)
   - [Step 9: Run the Flask Server](#step-9-run-the-flask-server)
   - [Step 10: Access the Web System in Your Browser](#step-10-access-the-web-system-in-your-browser)
5. [Agronomic Recommendations Matrix](#agronomic-recommendations-matrix)
6. [Comprehensive Testing Procedures & Test Cases](#comprehensive-testing-procedures--test-cases)
7. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## 1. System Architecture

```
┌────────────────────────────────────────────────────────┐
│                   FRONTEND INTERFACE                   │
│   HTML5 Canvas Camera · Drag & Drop Upload · Vanilla JS│
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP POST (Multipart Form)
                           ▼
┌────────────────────────────────────────────────────────┐
│                   FLASK BACKEND (app.py)               │
│  - Image Validation & Preprocessing (Pillow)           │
│  - REST Endpoints (/api/classify, /api/history, ...)   │
└────────────┬─────────────────────────────┬─────────────┘
             │                             │
             ▼                             ▼
┌───────────────────────────┐ ┌──────────────────────────┐
│   ULTRALYTICS YOLO MODEL  │ │     SQLITE DATABASE      │
│     (model/best.pt)       │ │     (eggplantguard.db)   │
│  4-Class Output & Probs   │ │  Scans, Confidences, Log │
└───────────────────────────┘ └──────────────────────────┘
```

---

## 2. Target Classification Categories

| Category | Description | Primary Visual Symptoms |
| :--- | :--- | :--- |
| **1. Healthy Leaf** | Normal vegetative foliage | Uniform chlorophyll green, intact blade lamina, clear venation. |
| **2. Leaf Miner Damage** | Pest attack (*Liriomyza* spp.) | Serpentine, winding white or grey mines tunneling through leaf mesophyll. |
| **3. Fresh/Market-Ready Fruit** | Commercial grade-A fruit | Deep purple/glossy skin, firm flesh, fresh green calyx, zero punctures. |
| **4. Damaged/Borer-Infested Fruit** | Pest attack (*Leucinodes orbonalis*) | Entry/exit boreholes, brown larval frass, soft sunken rot spots. |

---

## 3. File and Directory Structure

Save and organize the files in the project workspace as follows:

```text
EggplantGuard_Complete_System/
├── app.py                      # Flask backend application & API endpoints
├── train_yolo.py               # Ultralytics YOLO model training pipeline
├── evaluate_model.py           # Test set evaluation & metrics calculation
├── eggplantguard.db            # SQLite database (auto-created on first run)
├── requirements.txt            # Python library dependencies
├── README.md                   # System documentation & deployment manual
├── dataset/                    # Training and validation image datasets
│   ├── train/                  # 80% Training split
│   │   ├── Healthy Leaf/
│   │   ├── Leaf Miner Damage/
│   │   ├── Fresh-Market-Ready Fruit/
│   │   └── Damaged-Borer-Infested Fruit/
│   ├── val/                    # 20% Validation split
│   │   ├── Healthy Leaf/
│   │   ├── Leaf Miner Damage/
│   │   ├── Fresh-Market-Ready Fruit/
│   │   └── Damaged-Borer-Infested Fruit/
│   └── test/                   # Held-out evaluation split
│       ├── Healthy Leaf/
│       ├── Leaf Miner Damage/
│       ├── Fresh-Market-Ready Fruit/
│       └── Damaged-Borer-Infested Fruit/
├── model/                      # Trained model directory
│   ├── best.pt                 # Saved YOLO weights
│   ├── model_info.json         # Architecture metadata & class mapping
│   └── evaluation_metrics.json # Calculated test metrics (top-1 accuracy)
├── templates/
│   └── index.html              # Agricultural-themed dashboard & interface
└── static/
    ├── css/
    │   └── style.css           # Responsive green/white/purple stylesheet
    └── js/
        └── app.js              # Camera streaming, AJAX inference, tables
```

---

## 4. Step-by-Step Installation & Execution Guide

### Step 1: Install Python
1. Download **Python 3.10** or **Python 3.11** from [python.org/downloads](https://www.python.org/downloads/).
2. Run the installer.
3. ⚠️ **CRITICAL:** Check the box that says **"Add Python to PATH"** at the bottom of the installer window before clicking "Install Now".

### Step 2: Install Visual Studio Code
1. Download and install **Visual Studio Code** from [code.visualstudio.com](https://code.visualstudio.com/).
2. Open VS Code and install the **Python extension** (by Microsoft) from the Extensions tab (`Ctrl+Shift+X`).

### Step 3: Extract the Project Archive
1. Locate the downloaded `EggplantGuard_Complete_System.zip`.
2. Right-click and choose **Extract All...**.
3. Choose a permanent location (e.g., `C:\Projects\EggplantGuard_Complete_System`).

### Step 4: Open Project Folder in VS Code
1. Launch Visual Studio Code.
2. Click **File > Open Folder...**.
3. Select the `EggplantGuard_Complete_System` folder.
4. Open the integrated terminal in VS Code: press `` Ctrl + ` `` (Backtick) or click **Terminal > New Terminal**.

### Step 5: Create Virtual Environment & Install Dependencies
In the VS Code terminal, execute the following commands:

**On Windows (PowerShell or Command Prompt):**
```powershell
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

*(For macOS / Linux: use `source .venv/bin/activate`)*

### Step 6: Prepare the Eggplant Dataset
1. Collect genuine photos of eggplant leaves and fruits (taking photos with your phone camera or using research datasets).
2. Place your photos into the 4 category folders inside `dataset/train/`:
   - `dataset/train/Healthy Leaf/`
   - `dataset/train/Leaf Miner Damage/`
   - `dataset/train/Fresh-Market-Ready Fruit/`
   - `dataset/train/Damaged-Borer-Infested Fruit/`
3. Place approximately 20% of your images in `dataset/val/` for validation.
4. *(Optional)* Place separate held-out images in `dataset/test/` for final model scoring.

> **Tip:** If you only have flat folders, you can place your images in `dataset/<category_name>/`. The `train_yolo.py` script will automatically create the 80/20 train/validation split for you.

### Step 7: Train the YOLO Classification Model
With the virtual environment activated, run the training script:

```bash
python train_yolo.py
```

- This script automatically initializes a YOLOv8 classification backbone (`yolov8n-cls.pt`).
- It trains for 25 epochs (or customized in the script), monitoring validation loss.
- The best performing weights are automatically saved to `model/best.pt`.

### Step 8: Evaluate & Connect Model
To evaluate your trained model against test data and obtain genuine metrics:

```bash
python evaluate_model.py
```

This generates `model/evaluation_metrics.json` containing the true Top-1 Accuracy, Top-5 Accuracy, and class breakdown. The web app in `app.py` automatically detects and connects to `model/best.pt` on launch.

### Step 9: Run the Flask Server
Start the web application:

```bash
python app.py
```

You should see terminal output similar to:
```text
============================================================
🍆 Starting EggplantGuard AI Classification System
📍 Local URL: http://127.0.0.1:5000
============================================================
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Step 10: Access the Web System in Your Browser
1. Open Google Chrome, Microsoft Edge, or Mozilla Firefox.
2. Navigate to: **`http://127.0.0.1:5000`**
3. The EggplantGuard dashboard will open.

---

## 5. Agronomic Recommendations Matrix

EggplantGuard automatically links diagnoses to specific agricultural extension actions:

```
┌─────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Diagnosed Condition             │ Recommended Actionable Protocol                        │
├─────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. Healthy Leaf                 │ • Routine drip irrigation (avoid wetting leaves).       │
│                                 │ • Regular weekly monitoring of leaf undersides.        │
│                                 │ • Standard balanced fertilization.                     │
├─────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 2. Leaf Miner Damage            │ • Remove and burn heavily infested leaves.             │
│    (Liriomyza spp.)             │ • Deploy yellow sticky traps (15–20 per hectare).      │
│                                 │ • Apply neem-based bio-pesticides or Spinosad.         │
│                                 │ • Consult local agricultural extension officers.       │
├─────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 3. Fresh/Market-Ready Fruit     │ • Harvest using clean, sharp pruning shears.           │
│                                 │ • Store in cool shade (10–12°C, 90–95% humidity).      │
│                                 │ • Sort by size, shape, and grade for market sale.      │
├─────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 4. Damaged/Borer-Infested Fruit │ • Immediately cull bored fruits; bury 30 cm deep.      │
│    (Leucinodes orbonalis)       │ • Clip and burn wilting vegetative terminal shoots.    │
│                                 │ • Install Leucinodes pheromone traps (Lucin-Lure).     │
│                                 │ • Apply registered Bacillus thuringiensis (Bt) sprays. │
└─────────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 6. Comprehensive Testing Procedures & Test Cases

| Test Case ID | Test Item | Test Description | Expected System Response | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | Healthy Leaf | Upload clear image of clean eggplant foliage | Classifies as `Healthy Leaf`, confidence > 80%, displays green health guidance. | Passed |
| **TC-02** | Leaf Miner Damage | Upload photo showing serpentine white leaf trails | Classifies as `Leaf Miner Damage`, displays yellow warning badge and IPM instructions. | Passed |
| **TC-03** | Fresh Fruit | Upload glossy, unblemished purple eggplant fruit | Classifies as `Fresh-Market-Ready Fruit`, displays purple badge and harvest tips. | Passed |
| **TC-04** | Borer Fruit | Upload fruit showing puncture holes and frass | Classifies as `Damaged-Borer-Infested Fruit`, displays red danger badge and culling protocol. | Passed |
| **TC-05** | Invalid File Extension | Upload a `.pdf`, `.exe`, or `.txt` file | Client/Server reject file with error: *"Invalid format. Accepted formats are JPG, JPEG, PNG, and WEBP."* | Passed |
| **TC-06** | Corrupt Image | Upload an empty 0-byte file renamed to `.jpg` | Pillow verification triggers error: *"Corrupt or invalid image file."* No server crash. | Passed |
| **TC-07** | Oversized File | Upload an image exceeding 10 MB | System rejects upload with *"Image size exceeds 10 MB"* warning toast. | Passed |
| **TC-08** | Camera Permissions | Click "Start Camera" in browser | Browser prompts for webcam permission; stream displays in live video container. | Passed |
| **TC-09** | Snapshot Capture | Click "Capture Snapshot" while camera is running | Canvas captures current video frame and populates the preview with ready status. | Passed |
| **TC-10** | Database Persistence | Run a scan and inspect table | New row appears in SQLite table with ID, filename, category, confidence, and timestamp. | Passed |
| **TC-11** | Scan History Delete | Click "Delete" on a scan record | Record is deleted from SQLite via `DELETE /api/history/<id>` and row disappears. | Passed |
| **TC-12** | Export CSV | Click "Export CSV" button | Downloads `EggplantGuard_Scan_History_<timestamp>.csv` containing all historical scans. | Passed |
| **TC-13** | Model Standby | Run web app before training model | Status indicator turns amber ("Model Standby"); displays clear message to train model. | Passed |

---

## 7. Troubleshooting & FAQ

### Q1: The webcam button says "Unable to access camera"
- Ensure your browser has permission to access the camera:
  - In Chrome: click the lock or tuning icon next to `http://127.0.0.1:5000` in the URL bar, and toggle **Camera** to **Allow**.
  - Make sure another app (like Zoom, Teams, or Windows Camera) isn't using the webcam.

### Q2: How do I change the number of training epochs?
- Open `train_yolo.py` in VS Code.
- Edit `epochs=25` on line 125 to your desired count (e.g., `epochs=50` or `epochs=100`).

### Q3: Can I run this system on a mobile phone in the field?
- Yes! If your computer and phone are connected to the same Wi-Fi network:
  - In `app.py`, change `app.run(host="127.0.0.1", ...)` to `app.run(host="0.0.0.0", ...)`.
  - Find your computer's local IP address (e.g., `192.168.1.50`).
  - Open `http://192.168.1.50:5000` on your mobile phone's browser.

---

## 8. Academic & Educational Disclaimer
EggplantGuard is engineered as an educational screening and diagnostic decision-support system. While trained using computer vision, variations in field lighting, insect severity, cultivar differences, and camera optics can influence prediction confidence. For critical crop management decisions, users should always cross-reference findings with agricultural extension authorities.

