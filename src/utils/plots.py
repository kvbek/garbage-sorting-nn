import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pypalettes import load_cmap
from pyfonts import load_font

# Load custom fonts
poppins_light = load_font("https://github.com/google/fonts/blob/main/ofl/poppins/Poppins-Light.ttf?raw=true", danger_not_verify_ssl=True)
poppins_bold = load_font("https://github.com/google/fonts/blob/main/ofl/poppins/Poppins-Bold.ttf?raw=true", danger_not_verify_ssl=True)

# Font size
title_fontsize = 16
subtitle_fontsize = 13
tick_fontsize = 9
label_fontsize = 11

# Load color palettes
cmap = load_cmap("Semicossyphus_pulcher", reverse=True)
cmap_unreversed = load_cmap("Semicossyphus_pulcher", reverse=False)
cmap_x85 = load_cmap("X85", reverse=True)

def style_plot_fonts(ax, font_obj=poppins_light, tick_size=tick_fontsize, label_size=label_fontsize, remove_spines=True):
    """
    Applies custom font styling to a matplotlib axis.
    """
    # 1. Customize tick parameters (remove lengths, set base size)
    ax.tick_params(axis='both', labelsize=tick_size, length=0, rotation=0)

    # 2. Set font for tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font_obj)
        tick.set_fontsize(tick_size)
        
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_obj)
        tick.set_fontsize(tick_size)

    # 3. Set font for axis labels (only if they exist)
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontproperties=font_obj, fontsize=label_size)
    
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontproperties=font_obj, fontsize=label_size)

    if remove_spines:
        ax.spines[["top", "right"]].set_visible(False)


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