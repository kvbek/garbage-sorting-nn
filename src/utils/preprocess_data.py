# data_loader.py
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def get_garbage_datasets(data_dir, batch_size=64, img_size=(256, 256), model_type='custom'):
    """
    Loads and preprocesses the dataset.
    model_type can be 'custom' or 'mobilenet'
    """
    # 1. Load Raw Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=123,
        image_size=img_size, batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=123,
        image_size=img_size, batch_size=batch_size
    )
    
    # Extract class names before applying map functions
    class_names = train_ds.class_names

    # 2. Define Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # 3. Cache in RAM
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    # 4. Apply the correct preprocessing map
    if model_type == 'mobilenet':
        # MobileNet expects pixels between -1 and 1
        train_ds = train_ds.map(
            lambda x, y: (preprocess_input(data_augmentation(x, training=True)), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        val_ds = val_ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        # Custom CNN expects pixels between 0 and 1
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        val_ds = val_ds.map(
            lambda x, y: (x / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # 5. Prefetch for GPU/Neural Engine speed
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names