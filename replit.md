# Glaucoma Detection System

## Overview
An AI-powered Streamlit application for detecting glaucoma from fundus images using deep learning. The system includes 11 models (10 pre-trained + custom GlaucoNet) with binary classification, confidence-based severity estimation, and Grad-CAM visualization.

## Recent Changes
- **2026-01-23**: Initial project setup with complete application structure
  - Created multi-page Streamlit app with 5 pages (Home, Prediction, Comparison, Analytics, About)
  - Implemented custom GlaucoNet CNN architecture with residual connections and SE attention
  - Created training pipeline for all 11 models with two-stage fine-tuning
  - Added comprehensive evaluation metrics and Grad-CAM visualization
  - Implemented confidence-based severity level mapping

## Project Architecture

### Directory Structure
```
glaucoma_detection/
├── app.py                    # Main Streamlit entry point
├── train_models.py           # Model training script (run separately)
├── pages/                    # Streamlit multi-page structure
│   ├── 1_🏠_Home.py         # Home page with overview
│   ├── 2_🔬_Prediction.py   # Image upload and prediction
│   ├── 3_📊_Comparison.py   # Multi-model comparison
│   ├── 4_📈_Analytics.py    # Performance dashboard
│   └── 5_ℹ️_About.py        # Documentation and disclaimers
├── src/                      # Source modules
│   ├── __init__.py
│   ├── data_preprocessing.py # Data loading and augmentation
│   ├── custom_model.py       # GlaucoNet architecture
│   ├── evaluation.py         # Metrics and visualization
│   └── utils.py              # Shared utilities
├── saved_models/             # Trained model files (.h5)
├── results/                  # Training results
│   ├── plots/
│   └── metrics/
├── data/                     # Dataset directory
│   ├── raw/
│   └── processed/
└── .streamlit/config.toml    # Streamlit configuration
```

### Key Features
1. **Binary Classification**: Normal vs Glaucoma detection
2. **11 Models**: ResNet50, VGG16, VGG19, DenseNet121, DenseNet169, InceptionV3, Xception, MobileNetV2, EfficientNetB0, NASNetMobile, GlaucoNet
3. **Severity Estimation**: Derived from confidence scores (Normal, Borderline, Early, Moderate, Severe, Critical)
4. **Grad-CAM**: Visual explanations showing model attention regions
5. **Clinical Guidance**: CDR estimates and recommended actions

### Technical Stack
- **Framework**: TensorFlow/Keras
- **Frontend**: Streamlit (multi-page)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **ML**: scikit-learn, tf-keras-vis

### Running the Application
```bash
streamlit run app.py --server.port 5000
```

### Training Models
```bash
python train_models.py
```
Requires dataset in `Fundus_Scanes_Sorted/Train` and `Fundus_Scanes_Sorted/Validation`

### Model Input Sizes
- InceptionV3, Xception: 299x299
- All others: 224x224

### Severity Mapping
| Confidence | Severity | Est. CDR |
|------------|----------|----------|
| 0.0-0.3 | Normal | < 0.3 |
| 0.3-0.5 | Borderline | 0.3-0.5 |
| 0.5-0.7 | Early | 0.5-0.6 |
| 0.7-0.85 | Moderate | 0.6-0.7 |
| 0.85-0.95 | Severe | 0.7-0.9 |
| 0.95-1.0 | Critical | > 0.9 |

## User Preferences
- Clinical color scheme (white/blue background, dark blue headers)
- Medical disclaimers on all prediction outputs
- CDR values clearly marked as estimates, not measurements
