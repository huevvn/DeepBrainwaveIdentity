from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, log_dir='output/logs'):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = Path(log_dir) / f'{timestamp}.log'
        
    def log(self, msg):
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')
