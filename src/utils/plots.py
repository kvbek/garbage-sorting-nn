import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output

class LivePlotCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        # Initialize lists to store metrics
        self.epochs = []
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    def on_epoch_end(self, epoch, logs=None):
        # Store the metrics from the current epoch
        self.epochs.append(epoch + 1)
        for key in self.history.keys():
            self.history[key].append(logs.get(key))

        # Clear the previous chart to make room for the update
        clear_output(wait=True)
        
        # --- Your Custom Plotting Logic ---
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # Accuracy plot
        ax[0].plot(self.epochs, self.history['accuracy'], label='Train Accuracy', marker='.')
        ax[0].plot(self.epochs, self.history['val_accuracy'], label='Validation Accuracy', marker='.')
        ax[0].set_title('Model Accuracy')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].legend(loc='upper left')

        # Loss plot
        ax[1].plot(self.epochs, self.history['loss'], label='Train Loss', marker='.')
        ax[1].plot(self.epochs, self.history['val_loss'], label='Validation Loss', marker='.')
        ax[1].set_title('Model Loss')
        ax[1].set_ylabel('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].legend(loc='upper right')

        # Generic settings
        for a in ax:
            a.grid(color='lightgray', linestyle='--', linewidth=0.5)
            # a.set_xlim(1, max(self.epochs) if self.epochs else 1) # Optional: keep X-axis consistent
            for spine in ['top', 'right']:
                a.spines[spine].set_visible(False)
        
        plt.tight_layout()
        plt.show()