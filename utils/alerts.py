import smtplib
import json
import requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertSystem:
    def __init__(self, config):
        self.config = config
        self.alert_methods = []
        
        # Configure alert methods based on config
        if 'email' in config.get('alerts', {}):
            self.alert_methods.append(self.send_email_alert)
        if 'webhook' in config.get('alerts', {}):
            self.alert_methods.append(self.send_webhook_alert)
        if 'console' in config.get('alerts', {}):
            self.alert_methods.append(self.send_console_alert)
    
    def send_alert(self, alert_type, detection_results):
        """Send alert using all configured methods"""
        for method in self.alert_methods:
            try:
                method(alert_type, detection_results)
            except Exception as e:
                print(f"Failed to send alert via {method.__name__}: {e}")
    
    def send_email_alert(self, alert_type, results):
        """Send email alert"""
        email_config = self.config['alerts']['email']
        
        msg = MIMEMultipart()
        msg['From'] = email_config['from']
        msg['To'] = email_config['to']
        msg['Subject'] = f"Underwater Acoustic Alert: {alert_type}"
        
        body = f"""
Underwater Acoustic Detection Alert

Alert Type: {alert_type}
Timestamp: {results['timestamp']}
Detected Class: {results['predicted_class']}
Classification Confidence: {results['classification_confidence']:.3f}
Anomaly Status: {'Yes' if results['is_anomaly'] else 'No'}
Anomaly Confidence: {results['anomaly_confidence']:.3f}
Reconstruction Error: {results['reconstruction_error']:.6f}

This is an automated alert from your underwater acoustic monitoring system.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['username'], email_config['password'])
        
        text = msg.as_string()
        server.sendmail(email_config['from'], email_config['to'], text)
        server.quit()
        
        print(f"Email alert sent: {alert_type}")
    
    def send_webhook_alert(self, alert_type, results):
        """Send webhook alert (e.g., to Slack, Discord, or custom endpoint)"""
        webhook_config = self.config['alerts']['webhook']
        
        payload = {
            'alert_type': alert_type,
            'timestamp': results['timestamp'],
            'detection_results': results
        }
        
        if webhook_config['type'] == 'slack':
            slack_payload = {
                'text': f"🚨 Underwater Acoustic Alert: {alert_type}",
                'attachments': [{
                    'color': 'danger',
                    'fields': [
                        {'title': 'Detected Class', 'value': results['predicted_class'], 'short': True},
                        {'title': 'Confidence', 'value': f"{results['classification_confidence']:.3f}", 'short': True},
                        {'title': 'Anomaly', 'value': 'Yes' if results['is_anomaly'] else 'No', 'short': True},
                        {'title': 'Timestamp', 'value': results['timestamp'], 'short': False}
                    ]
                }]
            }
            
            response = requests.post(webhook_config['url'], json=slack_payload)
            
        else:  # Generic webhook
            response = requests.post(webhook_config['url'], json=payload)
        
        if response.status_code == 200:
            print(f"Webhook alert sent: {alert_type}")
        else:
            print(f"Failed to send webhook alert: {response.status_code}")
    
    def send_console_alert(self, alert_type, results):
        """Send console alert with formatted output"""
        print("\n" + "="*60)
        print(f"🚨 ACOUSTIC ALERT: {alert_type.upper()}")
        print("="*60)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Detected Class: {results['predicted_class']}")
        print(f"Classification Confidence: {results['classification_confidence']:.3f}")
        print(f"Anomaly Status: {'DETECTED' if results['is_anomaly'] else 'NORMAL'}")
        print(f"Anomaly Confidence: {results['anomaly_confidence']:.3f}")
        print(f"Reconstruction Error: {results['reconstruction_error']:.6f}")
        print("="*60 + "\n")