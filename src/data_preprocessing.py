import os
import shutil
import random
import numpy as np

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except (ImportError, ModuleNotFoundError):
    # Fallback if TensorFlow is not available
    class ImageDataGenerator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        
        def flow_from_directory(self, directory, **kwargs):
            return None

MODEL_INPUT_SIZES = {
    'ResNet50': (224, 224),
    'VGG16': (224, 224),
    'VGG19': (224, 224),
    'DenseNet121': (224, 224),
    'DenseNet169': (224, 224),
    'InceptionV3': (299, 299),
    'Xception': (299, 299),
    'MobileNetV2': (224, 224),
    'EfficientNetB0': (224, 224),
    'NASNetMobile': (224, 224),
    'GlaucoNet': (224, 224)
}

CLASS_NAMES = ['glaucoma', 'normal']

def get_input_size(model_name):
    return MODEL_INPUT_SIZES.get(model_name, (224, 224))

def create_train_generator(train_dir, target_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    return train_generator

def create_validation_generator(val_dir, target_size=(224, 224), batch_size=32):
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return val_generator

def create_test_generator(test_dir, target_size=(224, 224), batch_size=32):
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return test_generator

def split_validation_folder(validation_dir, test_dir, split_ratio=0.5, seed=42):
    random.seed(seed)
    
    if os.path.exists(test_dir):
        return
    
    os.makedirs(test_dir, exist_ok=True)
    
    for class_name in CLASS_NAMES:
        class_val_dir = os.path.join(validation_dir, class_name)
        class_test_dir = os.path.join(test_dir, class_name)
        
        if not os.path.exists(class_val_dir):
            continue
            
        os.makedirs(class_test_dir, exist_ok=True)
        
        images = os.listdir(class_val_dir)
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        test_images = images[:split_idx]
        
        for img in test_images:
            src = os.path.join(class_val_dir, img)
            dst = os.path.join(class_test_dir, img)
            shutil.move(src, dst)

def get_class_weights(train_generator):
    class_counts = np.bincount(train_generator.classes)
    total = sum(class_counts)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    return class_weights

def preprocess_single_image(image, target_size=(224, 224)):
    import cv2
    from PIL import Image
    
    if isinstance(image, str):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
    
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img

def apply_clahe(image):
    import cv2
    
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(image)
    
    return result

def extract_green_channel(image):
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return image[:, :, 1]
    return image

def get_dataset_stats(data_dir):
    stats = {}
    total_images = 0
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            stats[class_name] = count
            total_images += count
    
    stats['total'] = total_images
    return stats
