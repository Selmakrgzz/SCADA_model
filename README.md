# SCADA_model
Testing the Energy FaultDetector framework: https://aefdi.github.io/EnergyFaultDetector/index.html 

## Installation

### Prerequisites
- Python 3.11 or 3.12 (TensorFlow 2.18 and NumPy 1.26 do not support Python 3.13 yet)
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Selmakrgzz/SCADA_model.git
cd SCADA_model
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
```

macOS / Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs the [EnergyFaultDetector](https://aefdi.github.io/EnergyFaultDetector/index.html) framework along with TensorFlow, pandas, scikit-learn and Jupyter.

Model settings can be changed in `configs/config.yaml`.

## Project Structure

```
SCADA_model/
├── configs/
│   └── config.yaml            # EnergyFaultDetector config: preprocessing, autoencoder,
│                              # anomaly score, threshold selection, root cause analysis
├── notebooks/
│   ├── explore_data.ipynb     # Exploratory analysis of the SCADA data
│   ├── first_event.ipynb      # Walkthrough of fault detection on a first event
│   └── results_analysis.ipynb # Evaluation and visualization of model results
├── scripts/
│   └── run_farm_A.py          # Train/predict pipeline for Wind Farm A
├── data/                      # Raw SCADA datasets (not tracked, see .gitignore)
├── results/                   # Model outputs and plots (not tracked, see .gitignore)
├── .gitignore
├── requirements.txt           # Python dependencies (energy-fault-detector 0.8.1, TensorFlow 2.18)
└── README.md
```