
import os

class Reporter:
    def __init__(self, report_file):
        self.report_file = report_file
        self.cleared = False
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.report_file), exist_ok=True)
    
    def write(self, message):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not self.cleared:
            with open(self.report_file, "w") as f:
                f.write(f"[{timestamp}] INFO: Starting Enhanced Deep Embedded Clustering (DEC) Experiment - V2\n")
                f.write(f"[{timestamp}] INFO: Improvements:\n")
                f.write(f"[{timestamp}] INFO: - Enhanced feature extraction with multiple MFCC statistics\n")
                f.write(f"[{timestamp}] INFO: - Deeper network architecture with BatchNorm and Dropout\n")
                f.write(f"[{timestamp}] INFO: - Improved learning rates and training parameters\n")
                f.write(f"[{timestamp}] INFO: - Increased cluster count and latent dimensions\n")
            self.cleared = True
        
        with open(self.report_file, "a") as f:
            # Format message with timestamp and log level
            if "error" in message.lower() or "failed" in message.lower():
                log_level = "ERROR"
            elif "warning" in message.lower() or "warn" in message.lower():
                log_level = "WARN"
            else:
                log_level = "INFO"
            
            f.write(f"[{timestamp}] {log_level}: {message}\n")