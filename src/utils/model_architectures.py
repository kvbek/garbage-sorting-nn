from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf

def build_experimental_model(arch_type, input_shape=(256, 256, 3), num_classes=6, augmentation=True):
    tf.keras.backend.clear_session()
    
    inputs = layers.Input(shape=input_shape)
    x = inputs 
        
    # 2. Architecture Selection
    if arch_type == 'control':
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
    elif arch_type == 'shallow':
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
    elif arch_type == 'vgg_style':
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x) 
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x) 
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x) 
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
    elif arch_type == 'large_kernel':
        x = layers.Conv2D(32, (5, 5), padding='same')(x) 
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
    if arch_type == 'mobilenet_v2':
        # PREPROCESSING: Let Keras handle the strict -1 to 1 math here
        x = layers.Rescaling(scale=2.0, offset=-1.0)(x)

        
        # Load the frozen brain
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False 
        
        # Pass through with training=False to lock BatchNorm!
        x = base_model(x, training=False) 
    elif arch_type == 'mobilenet_v2_finetuned':
        x = layers.Rescaling(scale=2.0, offset=-1.0)(x)
        
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None # No need to download ImageNet weights again
        )
        base_model.trainable = False 
        x = base_model(x, training=False) 
        
        # Duplicate the head strictly for Phase 2
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Build the model immediately so we can load the weights
        model = models.Model(inputs=inputs, outputs=outputs, name=f'arch_{arch_type}')
        
        # 1. Load Phase 1 weights
        phase1_weights_path = '../models/mobilenet_v2_best.weights.h5'
        print(f"\n📥 Loading Phase 1 weights from: {phase1_weights_path}")
        model.load_weights(phase1_weights_path)
        
        # 2. Unfreeze the deep layers
        print("🔓 Unfreezing deep MobileNetV2 layers...")
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False
            
        # 3. Compile with Phase 2 Whisper Learning Rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 4. Return early to bypass the Phase 1 shared head below
        return model
    
    # 3. Shared Custom Decision Head (Automatically attaches to whichever model you picked above!)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # 4. Build and Compile
    model = models.Model(inputs=inputs, outputs=outputs, name=f'arch_{arch_type}')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model