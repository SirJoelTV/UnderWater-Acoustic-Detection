import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import yaml
import os
import joblib
from data.preprocessing import create_dataloaders
from models.cnn_lstm import MultiModalCNNLSTM
from utils.visualization import plot_confusion_matrix

def evaluate_model(data_dir, config, model_paths, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load label encoder
    label_encoder = joblib.load(model_paths['label_encoder'])

    # Create test dataloader only
    _, _, test_loader, _ = create_dataloaders(data_dir, config)

    # Load model
    model = MultiModalCNNLSTM(config)
    model.load_state_dict(torch.load(model_paths['classifier'], map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            mel_spec = batch['mel_spec'].to(device)
            mfcc = batch['mfcc'].to(device)
            lofar = batch['lofar'].to(device)
            labels = batch['label'].squeeze().to(device)

            outputs, _ = model(mel_spec, mfcc, lofar)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    target_names = label_encoder.classes_
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print("Classification Report:")
    print(report)

    # Save report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, target_names)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--model_dir', default='models/')
    parser.add_argument('--output_dir', default='output/')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_paths = {
        'classifier': os.path.join(args.model_dir, 'best_classifier.pth'),
        'label_encoder': os.path.join(args.model_dir, 'label_encoder.pkl')
    }

    evaluate_model(args.data_dir, config, model_paths, args.output_dir)
