import tensorflow as tf
from tensorflow.keras import layers, models
import os

def build_model(arch_type, input_shape=(256, 256, 3), num_classes=6):
    """Factory function that returns a compiled model based on the chosen architecture."""
    
    inputs = layers.Input(shape=input_shape)
    
    if arch_type == 'baseline':
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        
    elif arch_type == 'vgg_style':
        # Block 1: Two Convs
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        # Block 2: Two Convs
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        # Block 3: Two Convs
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
    elif arch_type == 'regularized':
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x) # Dropout in the vision layers!
        
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.GlobalAveragePooling2D()(x)

    # Common Decision Head for all architectures
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=f'arch_{arch_type}')
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
