#!/usr/bin/env python3
import sys
import pickle
import torch
from pathlib import Path
from time import time

sys.path.insert(0, 'src')
from config import *
from utils import Logger
from data_processing import process_dataset
from models import EEGNet, EarlyStopping
from training import Trainer, create_dataloaders
from visualization import plot_training_history, plot_performance_metrics, generate_report


def main():
    Path('output/models').mkdir(parents=True, exist_ok=True)
    Path('output/plots').mkdir(parents=True, exist_ok=True)
    
    log = Logger()
    log.log("\n" + "─" * 60)
    log.log("EEG Person Identification")
    log.log("─" * 60)
    
    # Load data
    log.log("\n[1/4] Data Loading")
    cache_path = OUTPUT_DIR / DATA_CACHE
    
    if cache_path.exists():
        log.log("  Loading cached data...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
    else:
        log.log(f"  Processing {NUM_SUBJECTS} subjects...")
        data = process_dataset(
            num_subjects=NUM_SUBJECTS,
            base_path=str(DATA_DIR),
            runs_per_subject=RUNS_PER_SUBJECT,
            max_segments=MAX_SEGMENTS
        )
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    X, y = data['X'], data['y']
    log.log(f"  Shape: {X.shape} | Subjects: {len(set(y))}")
    
    # Model
    log.log("\n[2/4] Model Setup")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader = create_dataloaders(X, y, batch_size=BATCH_SIZE, train_split=TRAIN_SPLIT)
    
    model = EEGNet(
        num_channels=X.shape[1],
        num_freq_bins=X.shape[2],
        num_time_steps=X.shape[3],
        num_classes=NUM_SUBJECTS,
        dropout_rate=DROPOUT_FC
    )
    
    trainer = Trainer(model, device=device, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, mode='min')
    
    log.log(f"  Device: {device} | Params: {model.count_parameters():,}")
    log.log(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    
    # Train
    log.log("\n[3/4] Training")
    start = time()
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=MAX_EPOCHS,
        early_stopping=early_stopping,
        save_path=str(MODEL_DIR / MODEL_FILE)
    )
    log.log(f"\n  Done in {time()-start:.1f}s")
    
    with open(OUTPUT_DIR / HISTORY_FILE, 'wb') as f:
        pickle.dump(history, f)
    
    # Evaluate
    log.log("\n[4/4] Evaluation")
    model.load_state_dict(torch.load(MODEL_DIR / MODEL_FILE))
    model = model.to(device)
    
    y_pred, y_true = trainer.evaluate(val_loader)
    
    with open(OUTPUT_DIR / PREDICTIONS_FILE, 'wb') as f:
        pickle.dump({'y_true': y_true, 'y_pred': y_pred}, f)
    
    # Outputs
    plot_training_history(history, save_path=str(PLOT_DIR / 'training.png'))
    plot_performance_metrics(y_true, y_pred, save_path=str(PLOT_DIR / 'performance.png'))
    report = generate_report(y_true, y_pred, history, save_path=str(OUTPUT_DIR / REPORT_FILE))
    
    log.log("\n" + "─" * 60)
    log.log(report)
    log.log("─" * 60)
    log.log(f"\nLog: {log.log_file}\n")


if __name__ == '__main__':
    main()
