import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import librosa.display

def plot_training_curves(train_losses, val_losses, val_accuracies):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curve
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_spectrogram(audio, sr, title="Spectrogram"):
    """Plot mel spectrogram"""
    plt.figure(figsize=(12, 4))
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_detection_timeline(detection_log):
    """Plot detection results over time"""
    timestamps = [r['timestamp'] for r in detection_log]
    confidences = [r['classification_confidence'] for r in detection_log]
    anomaly_scores = [r['anomaly_confidence'] for r in detection_log]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Classification confidence over time
    ax1.plot(timestamps[::10], confidences[::10], 'b-', alpha=0.7)  # Subsample for clarity
    ax1.set_ylabel('Classification Confidence')
    ax1.set_title('Classification Confidence Over Time')
    ax1.grid(True)
    
    # Anomaly scores over time
    ax2.plot(timestamps[::10], anomaly_scores[::10], 'r-', alpha=0.7)
    ax2.axhline(y=0.7, color='red', linestyle='--', label='Anomaly Threshold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Anomaly Score')
    ax2.set_title('Anomaly Detection Score Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig('detection_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()