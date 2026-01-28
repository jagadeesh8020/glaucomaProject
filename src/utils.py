import os
import numpy as np
from PIL import Image
import streamlit as st

SEVERITY_THRESHOLDS = {
    'normal': (0.0, 0.3),
    'borderline': (0.3, 0.5),
    'early': (0.5, 0.7),
    'moderate': (0.7, 0.85),
    'severe': (0.85, 0.95),
    'critical': (0.95, 1.0)
}

CDR_ESTIMATES = {
    'normal': '< 0.3',
    'borderline': '0.3 - 0.5',
    'early': '0.5 - 0.6',
    'moderate': '0.6 - 0.7',
    'severe': '0.7 - 0.9',
    'critical': '> 0.9'
}

RECOMMENDATIONS = {
    'normal': {
        'action': 'Regular eye checkup recommended',
        'urgency': 'Low',
        'color': '#10B981',
        'timeframe': 'Annual eye examination'
    },
    'borderline': {
        'action': 'Schedule an eye examination',
        'urgency': 'Moderate',
        'color': '#F59E0B',
        'timeframe': 'Within 3 months'
    },
    'early': {
        'action': 'Consult an ophthalmologist',
        'urgency': 'Moderate-High',
        'color': '#F97316',
        'timeframe': 'Within 1 month'
    },
    'moderate': {
        'action': 'Urgent ophthalmologist consultation',
        'urgency': 'High',
        'color': '#EF4444',
        'timeframe': 'Within 1 week'
    },
    'severe': {
        'action': 'Immediate medical attention required',
        'urgency': 'Very High',
        'color': '#DC2626',
        'timeframe': 'Within 1-2 days'
    },
    'critical': {
        'action': 'Emergency consultation required',
        'urgency': 'Critical',
        'color': '#991B1B',
        'timeframe': 'Immediately'
    }
}

MODEL_INFO = {
    'ResNet50': {
        'description': 'Deep residual network with 50 layers, known for skip connections that enable training very deep networks.',
        'params': '25.6M',
        'input_size': '224x224'
    },
    'VGG16': {
        'description': 'Classic architecture with 16 layers, known for its simplicity using only 3x3 convolutions.',
        'params': '138M',
        'input_size': '224x224'
    },
    'VGG19': {
        'description': 'Extended VGG architecture with 19 layers for improved feature extraction.',
        'params': '143M',
        'input_size': '224x224'
    },
    'DenseNet121': {
        'description': 'Dense connections between layers, where each layer receives inputs from all preceding layers.',
        'params': '8M',
        'input_size': '224x224'
    },
    'DenseNet169': {
        'description': 'Deeper DenseNet variant with 169 layers for enhanced feature reuse.',
        'params': '14M',
        'input_size': '224x224'
    },
    'InceptionV3': {
        'description': 'Uses inception modules with parallel convolutions of different sizes.',
        'params': '23.8M',
        'input_size': '299x299'
    },
    'Xception': {
        'description': 'Extreme inception with depthwise separable convolutions for efficiency.',
        'params': '22.9M',
        'input_size': '299x299'
    },
    'MobileNetV2': {
        'description': 'Lightweight architecture optimized for mobile devices with inverted residuals.',
        'params': '3.5M',
        'input_size': '224x224'
    },
    'EfficientNetB0': {
        'description': 'Balanced scaling of depth, width, and resolution for optimal performance.',
        'params': '5.3M',
        'input_size': '224x224'
    },
    'NASNetMobile': {
        'description': 'Neural Architecture Search optimized mobile network.',
        'params': '5.3M',
        'input_size': '224x224'
    },
    'GlaucoNet': {
        'description': 'Custom architecture with residual connections and squeeze-excitation attention for glaucoma detection.',
        'params': '~10M',
        'input_size': '224x224'
    }
}

def get_severity_level(confidence):
    for level, (low, high) in SEVERITY_THRESHOLDS.items():
        if low <= confidence < high:
            return level
    return 'critical' if confidence >= 0.95 else 'normal'

def get_severity_color(severity):
    return RECOMMENDATIONS.get(severity, {}).get('color', '#6B7280')

def get_recommendation(severity):
    return RECOMMENDATIONS.get(severity, RECOMMENDATIONS['normal'])

def get_cdr_estimate(severity):
    return CDR_ESTIMATES.get(severity, 'Unknown')

def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def preprocess_for_prediction(image, target_size=(224, 224)):
    img = image.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@st.cache_resource
def load_model(model_path):
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_available_models(models_dir='saved_models'):
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('.h5') or file.endswith('.keras'):
            model_name = file.replace('.h5', '').replace('.keras', '').replace('_best', '')
            models.append(model_name)
    
    return models

def format_confidence(confidence):
    return f"{confidence * 100:.1f}%"

def create_progress_bar_html(confidence, severity):
    color = get_severity_color(severity)
    percentage = confidence * 100
    
    html = f"""
    <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
        <div style="width: {percentage}%; background-color: {color}; padding: 10px 0; text-align: center; color: white; font-weight: bold; border-radius: 10px;">
            {percentage:.1f}%
        </div>
    </div>
    """
    return html

def get_clinical_notes():
    return """
    **Important Clinical Considerations:**
    
    - **Intraocular Pressure (IOP):** Normal range is 10-21 mmHg. Elevated IOP is a major risk factor for glaucoma.
    
    - **Visual Field Testing:** Recommended for comprehensive glaucoma assessment to detect any peripheral vision loss.
    
    - **OCT Imaging:** Optical Coherence Tomography provides detailed imaging of the optic nerve and retinal nerve fiber layer.
    
    - **Family History:** Glaucoma has a genetic component. Patients with family history are at higher risk.
    
    - **Treatment Options:**
      - Eye drops to reduce IOP
      - Laser therapy
      - Surgical procedures (trabeculectomy, drainage implants)
    """

def get_disclaimer():
    return """
    **Medical Disclaimer:**
    
    This tool is for educational and research purposes only. The predictions made by this system 
    should NOT be used as a substitute for professional medical diagnosis. 
    
    The Cup-to-Disc Ratio (CDR) values displayed are **estimates** derived from the model's 
    confidence scores and are NOT actual measurements from the fundus image.
    
    Always consult a qualified ophthalmologist for proper diagnosis and treatment of eye conditions.
    """

SAMPLE_METRICS = {
    'ResNet50': {'accuracy': 0.92, 'precision': 0.91, 'recall': 0.93, 'f1_score': 0.92, 'roc_auc': 0.96, 'sensitivity': 0.93, 'specificity': 0.91},
    'VGG16': {'accuracy': 0.89, 'precision': 0.88, 'recall': 0.90, 'f1_score': 0.89, 'roc_auc': 0.94, 'sensitivity': 0.90, 'specificity': 0.88},
    'VGG19': {'accuracy': 0.88, 'precision': 0.87, 'recall': 0.89, 'f1_score': 0.88, 'roc_auc': 0.93, 'sensitivity': 0.89, 'specificity': 0.87},
    'DenseNet121': {'accuracy': 0.93, 'precision': 0.92, 'recall': 0.94, 'f1_score': 0.93, 'roc_auc': 0.97, 'sensitivity': 0.94, 'specificity': 0.92},
    'DenseNet169': {'accuracy': 0.94, 'precision': 0.93, 'recall': 0.95, 'f1_score': 0.94, 'roc_auc': 0.97, 'sensitivity': 0.95, 'specificity': 0.93},
    'InceptionV3': {'accuracy': 0.91, 'precision': 0.90, 'recall': 0.92, 'f1_score': 0.91, 'roc_auc': 0.95, 'sensitivity': 0.92, 'specificity': 0.90},
    'Xception': {'accuracy': 0.92, 'precision': 0.91, 'recall': 0.93, 'f1_score': 0.92, 'roc_auc': 0.96, 'sensitivity': 0.93, 'specificity': 0.91},
    'MobileNetV2': {'accuracy': 0.87, 'precision': 0.86, 'recall': 0.88, 'f1_score': 0.87, 'roc_auc': 0.92, 'sensitivity': 0.88, 'specificity': 0.86},
    'EfficientNetB0': {'accuracy': 0.93, 'precision': 0.92, 'recall': 0.94, 'f1_score': 0.93, 'roc_auc': 0.97, 'sensitivity': 0.94, 'specificity': 0.92},
    'NASNetMobile': {'accuracy': 0.90, 'precision': 0.89, 'recall': 0.91, 'f1_score': 0.90, 'roc_auc': 0.94, 'sensitivity': 0.91, 'specificity': 0.89},
    'GlaucoNet': {'accuracy': 0.91, 'precision': 0.90, 'recall': 0.92, 'f1_score': 0.91, 'roc_auc': 0.95, 'sensitivity': 0.92, 'specificity': 0.90}
}
