import torch
import torchaudio
import pyaudio
import numpy as np
import queue
import threading
import time
import yaml
import joblib
from datetime import datetime
import json

from models.cnn_lstm import MultiModalCNNLSTM
from models.autoencoder import ConvolutionalAutoEncoder, AnomalyDetector
from utils.audio_utils import AudioProcessor
from utils.alerts import AlertSystem

class RealTimeDetector:
    def __init__(self, config_path, model_paths):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(self.config)
        
        # Load models
        self.classifier = MultiModalCNNLSTM(self.config).to(self.device)
        self.classifier.load_state_dict(torch.load(model_paths['classifier'], map_location=self.device))
        self.classifier.eval()
        
        self.autoencoder = ConvolutionalAutoEncoder(self.config).to(self.device)
        self.autoencoder.load_state_dict(torch.load(model_paths['autoencoder'], map_location=self.device))
        self.autoencoder.eval()
        
        # Load label encoder
        self.label_encoder = joblib.load(model_paths['label_encoder'])
        
        # Initialize anomaly detector
        self.anomaly_detector = AnomalyDetector(self.autoencoder)
        # Set a default threshold (should be fitted on normal data)
        self.anomaly_detector.threshold = 0.5
        
        # Audio streaming setup
        self.sample_rate = self.config['data']['sample_rate']
        self.buffer_size = self.config['realtime']['buffer_size']
        self.audio_queue = queue.Queue()
        
        # Alert system
        self.alert_system = AlertSystem(self.config)
        
        # Detection thresholds
        self.anomaly_threshold = self.config['realtime']['threshold_anomaly']
        self.confidence_threshold = self.config['realtime']['threshold_confidence']
        self.alert_cooldown = self.config['realtime']['alert_cooldown']
        
        # State management
        self.is_running = False
        self.last_alert_time = {}
        
        # Results logging
        self.detection_log = []
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio stream status: {status}")
            
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def start_audio_stream(self):
        """Start audio input stream"""
        self.audio = pyaudio.PyAudio()
        
        # Find the best input device (preferably USB audio interface for hydrophone)
        self.input_device = None
        for i in range(self.audio.get_device_count()):
            dev_info = self.audio.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                print(f"Audio device {i}: {dev_info['name']}")
                if 'usb' in dev_info['name'].lower() or 'line' in dev_info['name'].lower():
                    self.input_device = i
        
        if self.input_device is None:
            self.input_device = self.audio.get_default_input_device_info()['index']
            
        print(f"Using audio device: {self.input_device}")
        
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=self.buffer_size,
            stream_callback=self.audio_callback
        )
        
        self.stream.start_stream()
        
    def process_audio_segment(self, audio_segment):
        """Process a single audio segment and return classification results"""
        try:
            # Preprocess audio
            processed_audio = self.audio_processor.preprocess_audio(audio_segment)
            features = self.audio_processor.extract_features(processed_audio)
            
            # Convert to tensors
            mel_spec = torch.FloatTensor(features['mel_spec']).unsqueeze(0).to(self.device)
            mfcc = torch.FloatTensor(features['mfcc']).unsqueeze(0).to(self.device)
            lofar = torch.FloatTensor(features['lofar']).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Classification
                class_output, features_vec = self.classifier(mel_spec, mfcc, lofar)
                class_probabilities = torch.softmax(class_output, dim=1)
                predicted_class = torch.argmax(class_probabilities, dim=1).item()
                confidence = class_probabilities[0, predicted_class].item()
                
                # Anomaly detection
                is_anomaly, anomaly_confidence, reconstruction_error = self.anomaly_detector.detect_anomaly(mel_spec)
                
                # Decode class label
                class_label = self.label_encoder.inverse_transform([predicted_class])[0]
                
                results = {
                    'timestamp': datetime.now().isoformat(),
                    'predicted_class': class_label,
                    'classification_confidence': float(confidence),
                    'is_anomaly': bool(is_anomaly[0]),
                    'anomaly_confidence': float(anomaly_confidence[0]),
                    'reconstruction_error': float(reconstruction_error[0])
                }
                
                return results
                
        except Exception as e:
            print(f"Error processing audio segment: {e}")
            return None
    
    def detection_loop(self):
        """Main detection loop"""
        audio_buffer = np.array([])
        segment_length = int(self.sample_rate * self.config['data']['duration'])
        
        while self.is_running:
            try:
                # Get audio data from queue
                if not self.audio_queue.empty():
                    new_audio = self.audio_queue.get()
                    audio_buffer = np.concatenate([audio_buffer, new_audio])
                    
                    # Process when we have enough audio
                    if len(audio_buffer) >= segment_length:
                        # Extract segment
                        segment = audio_buffer[:segment_length]
                        audio_buffer = audio_buffer[segment_length//2:]  # 50% overlap
                        
                        # Process segment
                        results = self.process_audio_segment(segment)
                        
                        if results:
                            self.handle_detection_results(results)
                            
                else:
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
                    
            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(1)
    
    def handle_detection_results(self, results):
        """Handle detection results and trigger alerts if necessary"""
        # Log results
        self.detection_log.append(results)
        
        # Print real-time results
        print(f"\n[{results['timestamp']}]")
        print(f"Predicted: {results['predicted_class']} (Confidence: {results['classification_confidence']:.3f})")
        
        if results['is_anomaly']:
            print(f"⚠️ ANOMALY DETECTED! (Score: {results['anomaly_confidence']:.3f})")
        
        # Trigger alerts
        should_alert = False
        alert_type = None
        
        # High-confidence classification alert
        if results['classification_confidence'] > self.confidence_threshold:
            class_type = results['predicted_class'].split('_')[0]
            if class_type in ['ships', 'anomaly']:
                should_alert = True
                alert_type = f"High confidence {class_type} detection"
        
        # Anomaly alert
        if results['is_anomaly'] and results['anomaly_confidence'] > self.anomaly_threshold:
            should_alert = True
            alert_type = "Acoustic anomaly detected"
        
        # Check cooldown
        if should_alert:
            current_time = time.time()
            last_alert = self.last_alert_time.get(alert_type, 0)
            
            if current_time - last_alert > self.alert_cooldown:
                self.alert_system.send_alert(alert_type, results)
                self.last_alert_time[alert_type] = current_time
    
    def start(self):
        """Start real-time detection"""
        print("Starting real-time underwater acoustic detection...")
        
        self.is_running = True
        
        # Start audio stream
        self.start_audio_stream()
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        print("Detection system running. Press Ctrl+C to stop.")
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping detection system...")
            self.stop()
    
    def stop(self):
        """Stop real-time detection"""
        self.is_running = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
        
        # Save detection log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"detection_log_{timestamp}.json"
        
        with open(log_filename, 'w') as f:
            json.dump(self.detection_log, f, indent=2)
            
        print(f"Detection log saved to {log_filename}")
        print("Detection system stopped.")

# Usage example
if __name__ == '__main__':
    config_path = 'config/config.yaml'
    model_paths = {
        'classifier': 'best_classifier.pth',
        'autoencoder': 'best_autoencoder.pth',
        'label_encoder': 'label_encoder.pkl'
    }
    
    detector = RealTimeDetector(config_path, model_paths)
    detector.start()