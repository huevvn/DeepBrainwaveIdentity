from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'eegDataset'
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODEL_DIR = OUTPUT_DIR / 'models'
LOG_DIR = OUTPUT_DIR / 'logs'
PLOT_DIR = OUTPUT_DIR / 'plots'

NUM_SUBJECTS = 109
RUNS_PER_SUBJECT = 6
MAX_SEGMENTS = 8
LOWCUT_FREQ = 1.0
HIGHCUT_FREQ = 40.0

DROPOUT_SPATIAL = 0.2
DROPOUT_FC = 0.4
CNN_CHANNELS = [32, 64]
LSTM_HIDDEN = 512
LSTM_LAYERS = 2

BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-5
MAX_EPOCHS = 75
TRAIN_SPLIT = 0.8

EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.001

LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3

DATA_CACHE = 'data.pkl'           # Preprocessed EEG features
MODEL_FILE = 'model.pth'          # Best model weights
HISTORY_FILE = 'history.pkl'      # Training metrics
PREDICTIONS_FILE = 'preds.pkl'    # Validation predictions
REPORT_FILE = 'report.txt'        # Performance summary
