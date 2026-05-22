# src/config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Hyperparameters
IMG_SIZE = (224, 224) # MobileNetV2 default
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Classes
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']