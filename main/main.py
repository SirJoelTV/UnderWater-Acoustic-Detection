#!/usr/bin/env python3
"""
Underwater Acoustic Anomaly Detection System
Main application entry point
"""

import argparse
import yaml
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Underwater Acoustic Anomaly Detection System')
    parser.add_argument('mode', choices=['train', 'detect', 'evaluate'], 
                       help='Operating mode')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--data_dir', help='Path to training data directory')
    parser.add_argument('--model_dir', default='models/',
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', default='output/',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'train':
        from training.train import main as train_main
        import sys
        
        # Override sys.argv for training script
        sys.argv = [
            'train.py',
            '--config', args.config,
            '--data_dir', args.data_dir
        ]
        train_main()
        
    elif args.mode == 'detect':
        from inference.real_time_detector import RealTimeDetector
        
        model_paths = {
            'classifier': os.path.join(args.model_dir, 'best_classifier.pth'),
            'autoencoder': os.path.join(args.model_dir, 'best_autoencoder.pth'),
            'label_encoder': os.path.join(args.model_dir, 'label_encoder.pkl')
        }
        
        detector = RealTimeDetector(args.config, model_paths)
        detector.start()
        
    elif args.mode == 'evaluate':
        from training.evaluate import evaluate_model
        
        model_paths = {
            'classifier': os.path.join(args.model_dir, 'best_classifier.pth'),
            'autoencoder': os.path.join(args.model_dir, 'best_autoencoder.pth'),
            'label_encoder': os.path.join(args.model_dir, 'label_encoder.pkl')
        }
        
        evaluate_model(args.data_dir, config, model_paths, args.output_dir)

if __name__ == '__main__':
    main()