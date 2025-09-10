import librosa
import numpy as np
import soundfile as sf
import torch
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class AudioProcessor:
    def __init__(self, config):
        self.sr = config['data']['sample_rate']
        self.window_size = config['data']['window_size']
        self.hop_length = config['data']['hop_length']
        self.n_mels = config['data']['n_mels']
        self.n_mfcc = config['data']['n_mfcc']
        self.duration = config['data']['duration']
        
    def load_audio(self, file_path, start_time=None, duration=None):
        """Load audio file with optional time slicing"""
        try:
            audio, sr = librosa.load(
                file_path, 
                sr=self.sr,
                offset=start_time,
                duration=duration or self.duration
            )
            return audio, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def preprocess_audio(self, audio):
        """Apply preprocessing filters"""
        # High-pass filter to remove very low frequencies
        b, a = signal.butter(4, 50, btype='high', fs=self.sr)
        audio = signal.filtfilt(b, a, audio)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Ensure fixed length
        target_length = int(self.sr * self.duration)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            
        return audio
    
    def extract_features(self, audio):
        """Extract multiple acoustic features"""
        features = {}
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=self.n_mels,
            hop_length=self.hop_length, n_fft=self.window_size
        )
        features['mel_spec'] = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
            hop_length=self.hop_length, n_fft=self.window_size
        )
        features['mfcc'] = mfcc
        
        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )
        
        # LOFAR spectrum (specialized for underwater acoustics)
        features['lofar'] = self.compute_lofar(audio)
        
        return features
    
    def compute_lofar(self, audio):
        """Compute LOFAR spectrum for underwater acoustics"""
        # STFT
        stft = librosa.stft(audio, n_fft=self.window_size, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Log compression
        lofar = np.log10(magnitude + 1e-10)
        
        return lofar
    
    def segment_audio(self, audio, segment_duration=2.0, overlap=0.5):
        """Segment long audio into smaller chunks"""
        segment_samples = int(segment_duration * self.sr)
        overlap_samples = int(overlap * segment_samples)
        step = segment_samples - overlap_samples
        
        segments = []
        for i in range(0, len(audio) - segment_samples + 1, step):
            segment = audio[i:i + segment_samples]
            segments.append(segment)
            
        return segments
