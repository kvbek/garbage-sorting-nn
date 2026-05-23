# src/config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Hyperparameters
IMG_SIZE = (256, 256)
IMG_SIZE_NET_V2 = (224, 224) # MobileNetV2 default
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 42

# Classes
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']