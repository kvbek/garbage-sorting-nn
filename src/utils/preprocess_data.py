# utils/preprocess_data.py
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def get_garbage_datasets(data_dir, batch_size=64, seed=42, img_size=(256, 256), model_type='custom'):
    """
    Loads and preprocesses the dataset.
    model_type can be 'custom' or 'mobilenet'
    """
    # 1. Load Raw Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, 
        validation_split=0.2, 
        subset="training", 
        seed=seed,
        image_size=img_size, 
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, 
        validation_split=0.2, 
        subset="validation", 
        seed=seed,
        image_size=img_size, 
        batch_size=batch_size
    )
    
    # Extract class names before applying map functions
    class_names = train_ds.class_names

    # 2. Define Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # 3. Cache in RAM and Shuffle (Crucial for preventing sequence memorization)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(buffer_size=1000) # Added shuffle here!
    val_ds = val_ds.cache() # Never shuffle validation data

    # 4. Apply the correct preprocessing map (ONLY MATH, NO AUGMENTATION)
    if model_type == 'mobilenet':
        train_ds = train_ds.map(
            lambda x, y: (preprocess_input(x), y), # Removed augmentation
            num_parallel_calls=tf.data.AUTOTUNE
        )
        val_ds = val_ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        train_ds = train_ds.map(
            lambda x, y: (x / 255.0, y), # Removed augmentation
            num_parallel_calls=tf.data.AUTOTUNE
        )
        val_ds = val_ds.map(
            lambda x, y: (x / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # 5. Prefetch for GPU/Neural Engine speed (Always the very last step)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names