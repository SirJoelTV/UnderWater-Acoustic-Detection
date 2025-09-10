import os
import torch
import yaml
import joblib
from tqdm import tqdm
from models.cnn_lstm import MultiModalCNNLSTM
from models.autoencoder import ConvolutionalAutoEncoder, AnomalyDetector
from utils.audio_utils import AudioProcessor
import json
from datetime import datetime

class BatchProcessor:
    def __init__(self, config_path, model_paths):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.audio_processor = AudioProcessor(self.config)

        self.classifier = MultiModalCNNLSTM(self.config).to(self.device)
        self.classifier.load_state_dict(torch.load(model_paths['classifier'], map_location=self.device))
        self.classifier.eval()

        self.autoencoder = ConvolutionalAutoEncoder(self.config).to(self.device)
        self.autoencoder.load_state_dict(torch.load(model_paths['autoencoder'], map_location=self.device))
        self.autoencoder.eval()

        self.label_encoder = joblib.load(model_paths['label_encoder'])
        self.anomaly_detector = AnomalyDetector(self.autoencoder)
        self.anomaly_detector.threshold = 0.5  # Should be set after fitting during training

    def process_file(self, file_path):
        audio, _ = self.audio_processor.load_audio(file_path, duration=self.config['data']['duration'])
        if audio is None:
            return None

        audio = self.audio_processor.preprocess_audio(audio)
        features = self.audio_processor.extract_features(audio)

        mel_spec = torch.FloatTensor(features['mel_spec']).unsqueeze(0).to(self.device)
        mfcc = torch.FloatTensor(features['mfcc']).unsqueeze(0).to(self.device)
        lofar = torch.FloatTensor(features['lofar']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            class_output, _ = self.classifier(mel_spec, mfcc, lofar)
            class_prob = torch.softmax(class_output, dim=1)
            pred_class = torch.argmax(class_prob, dim=1).item()
            confidence = class_prob[0, pred_class].item()

            is_anomaly, anomaly_confidence, recon_error = self.anomaly_detector.detect_anomaly(mel_spec)

        result = {
            "file_path": file_path,
            "timestamp": datetime.now().isoformat(),
            "predicted_class": self.label_encoder.inverse_transform([pred_class])[0],
            "classification_confidence": confidence,
            "is_anomaly": bool(is_anomaly[0]),
            "anomaly_confidence": anomaly_confidence[0].item(),
            "reconstruction_error": recon_error[0].item()
        }
        return result

    def process_directory(self, dir_path, output_file='batch_results.json'):
        results = []
        for root, _, files in os.walk(dir_path):
            for f in tqdm(files):
                if f.endswith(('.wav', '.flac', '.mp3')):
                    file_path = os.path.join(root, f)
                    r = self.process_file(file_path)
                    if r:
                        results.append(r)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Batch processing complete. Results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--model_dir', default='models/')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_file', default='batch_results.json')
    args = parser.parse_args()

    model_paths = {
        'classifier': os.path.join(args.model_dir, 'best_classifier.pth'),
        'autoencoder': os.path.join(args.model_dir, 'best_autoencoder.pth'),
        'label_encoder': os.path.join(args.model_dir, 'label_encoder.pkl')
    }

    processor = BatchProcessor(args.config, model_paths)
    processor.process_directory(args.input_dir, args.output_file)
