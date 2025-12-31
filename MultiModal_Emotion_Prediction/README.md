# MultiModal Emotion Prediction

A deep learning project for emotion recognition using multimodal physiological signals from the WESAD dataset. This project combines CNN and Transformer architectures with behavioral features for emotion classification.

## Project Overview

This project processes wearable sensor data (electrodermal activity, blood volume pulse, acceleration, and temperature) to predict emotional states. The pipeline includes:

1. **Data Exploration** - Understanding WESAD dataset structure
2. **Signal Alignment & Windowing** - Filtering, resampling, and segmenting signals
3. **Feature Engineering** - Extracting statistical, temporal, and behavioral features
4. **Deep Learning Dataset Preparation** - Creating PyTorch DataLoaders
5. **CNN + Transformer Fusion** - Training multimodal deep learning models

## Dataset

This project uses the **WESAD (Wearable Stress and Affect Detection)** dataset, which contains:
- **Chest and wrist sensors** capturing physiological signals
- **Multiple subjects** (S2, S3, S4, ..., S17)
- **Emotion labels**: Baseline (1), Stress (2), Amusement (3)
- **Signals**: EDA (4 Hz), BVP (64 Hz), Temperature (4 Hz), Acceleration (32 Hz)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MultiModal_Emotion_Prediction
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   # Using venv
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n emotion_pred python=3.9
   conda activate emotion_pred
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
MultiModal_Emotion_Prediction/
├── src/
│   ├── 01_exploration.ipynb              # Data exploration
│   ├── 02_Signal_Alignment_WESAD_w60.ipynb  # Signal preprocessing
│   ├── 03_feature_engineering.ipynb      # Feature extraction
│   ├── 04_dl_dataset.ipynb               # Dataset preparation
│   ├── 05_CNN_TRANSFORMER_FUSION.ipynb   # Model training
│   ├── aligned_windows_*.npz             # Preprocessed signals
│   └── features_*.npz                    # Engineered features
├── data/                                  # WESAD dataset (not included)
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## Usage Guide

### 1. Data Preparation

Ensure your WESAD dataset is organized as:
```
D:/Dissertation/Data/WESAD/
├── S2/S2.pkl
├── S3/S3.pkl
└── ... (other subjects)
```

### 2. Run the Pipeline

Execute the notebooks in order:

#### Step 1: Data Exploration
```bash
jupyter notebook src/01_exploration.ipynb
```
- Inspects WESAD dataset structure
- Verifies signal shapes and sampling rates

#### Step 2: Signal Alignment & Windowing
```bash
jupyter notebook src/02_Signal_Alignment_WESAD_w60.ipynb
```
- Applies Butterworth filters (EDA, Temperature, BVP, Acceleration)
- Resamples all signals to 4 Hz
- Performs correct label downsampling (700 Hz → 4 Hz)
- Creates 60-sample sliding windows
- Outputs: `aligned_windows_S{subject}_w60.npz`

#### Step 3: Feature Engineering
```bash
jupyter notebook src/03_feature_engineering.ipynb
```
- Extracts statistical features (mean, std, min, max, skewness, kurtosis)
- Computes behavioral features from acceleration data
- Outputs: `features_S{subject}_w60.npz`

#### Step 4: Deep Learning Dataset Preparation
```bash
jupyter notebook src/04_dl_dataset.ipynb
```
- Normalizes features using z-score normalization
- Creates PyTorch Dataset and DataLoaders
- Splits data into train/test sets (80/20)

#### Step 5: Model Training & Evaluation
```bash
jupyter notebook src/05_CNN_TRANSFORMER_FUSION.ipynb
```
- Trains CNN + Temporal Transformer + Behavioral Fusion model
- Single-subject training with 30 epochs
- Multi-subject training across all subjects
- Evaluates with classification metrics and confusion matrix

### 3. Model Architecture

The emotion prediction model consists of:

- **CNN Encoders**: Process each physiological signal independently
  - Conv1d layers: 1 → 32 → 128 channels
  - Batch normalization and ReLU activations
  
- **Temporal Transformer**: Captures temporal dependencies
  - TransformerEncoder with 4 attention heads
  - 1 encoder layer for efficiency

- **Behavior Encoder**: Processes acceleration-derived features
  - Fully connected layers: Input → 128 → 128 dimensions
  
- **Fusion Transformer**: Combines multimodal representations
  - Fuses features from all 5 encoders (EDA, BVP, ACC, Temp, Behavior)
  - TransformerEncoder for cross-modal attention

- **Classifier**: Final prediction head
  - Linear layers with dropout (0.4)
  - 3-way classification (Baseline, Stress, Amusement)

## Key Features

- **Multimodal Fusion**: Combines multiple physiological modalities
- **Temporal Modeling**: Transformer architecture for sequence modeling
- **Balanced Learning**: Weighted CrossEntropyLoss for class imbalance
- **Progressive Training**: Frozen encoders initially, unfrozen after 10 epochs
- **Multi-subject Support**: Trains on data from multiple subjects

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21.0 | Numerical computations |
| scipy | ≥1.7.0 | Signal processing (filters, resampling) |
| scikit-learn | ≥1.0.0 | Machine learning utilities (splits, metrics, class weights) |
| matplotlib | ≥3.5.0 | Data visualization |
| torch | ≥1.10.0 | Deep learning framework |
| jupyter | ≥1.0.0 | Interactive notebook environment |
| ipython | ≥7.0.0 | Enhanced Python shell |
| pandas | ≥1.3.0 | Data manipulation |

## Configuration

Edit the following parameters in each notebook:

- **DATA_ROOT**: Path to WESAD dataset
- **SUBJECTS**: List of subject IDs to process
- **WINDOW_SIZE**: Temporal window length (default: 60)
- **OVERLAP**: Window overlap (default: 30)
- **BATCH_SIZE**: Training batch size (default: 32)
- **EPOCHS**: Number of training epochs (default: 30)
- **DEVICE**: GPU/CPU selection (default: auto-detect)

## Output Files

After running the pipeline:

- `aligned_windows_S{id}_w60.npz`: Preprocessed windowed signals
- `features_S{id}_w60.npz`: Engineered features with labels
- `aligned_windows_ALL_w60.npz`: Concatenated multi-subject data
- `WESAD_ALL_w60.npz`: Full dataset for training

## Expected Performance

The model achieves emotion classification on the WESAD dataset with:
- **Input**: Physiological signals (4 dimensions) + Behavioral features
- **Output**: 3-class emotion prediction (Baseline, Stress, Amusement)
- **Evaluation**: Classification report with precision, recall, F1-score

## Troubleshooting

### Issue: FileNotFoundError for WESAD data
**Solution**: Verify WESAD dataset path in notebooks matches your local setup
```python
DATA_ROOT = "D:/Dissertation/Data/WESAD"  # Update this path
```

### Issue: GPU out of memory
**Solution**: Reduce batch size in training cells
```python
BATCH_SIZE = 16  # Reduce from 32
```

### Issue: Missing .npz files in Step-4/5
**Solution**: Ensure all previous steps ran successfully
```bash
# Check for outputs from previous steps
ls src/aligned_windows_*.npz
ls src/features_*.npz
```

## References

- WESAD Dataset: https://ubicomp.ethz.ch/research/publications/2018/wesad/
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- SciPy Signal Processing: https://docs.scipy.org/doc/scipy/reference/signal.html

## Notes

- All physiological signals are downsampled to 4 Hz for computational efficiency
- Labels are correctly downsampled from 700 Hz using mode aggregation
- Features are normalized per-window to account for individual differences
- The model uses transfer learning principles with initial encoder freezing

## License

See LICENSE file for details.


