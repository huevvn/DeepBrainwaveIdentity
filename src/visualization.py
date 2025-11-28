import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_training_history(history, save_path='output/plots/training.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['train_loss'], 'o-', label='Training', 
                    linewidth=2, markersize=4, alpha=0.8)
    axes[0, 0].plot(epochs, history['val_loss'], 's-', label='Validation', 
                    linewidth=2, markersize=4, alpha=0.8)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(frameon=True, shadow=True)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['train_acc'], 'o-', label='Training', 
                    linewidth=2, markersize=4, alpha=0.8)
    axes[0, 1].plot(epochs, history['val_acc'], 's-', label='Validation', 
                    linewidth=2, markersize=4, alpha=0.8)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend(frameon=True, shadow=True)
    axes[0, 1].grid(True, alpha=0.3)
    
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 0].plot(epochs, loss_diff, 'o-', linewidth=2, markersize=4, 
                    color='coral', alpha=0.8)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].fill_between(epochs, 0, loss_diff, where=(loss_diff > 0), 
                            alpha=0.3, color='red', label='Overfitting region')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Val Loss - Train Loss', fontsize=11)
    axes[1, 0].set_title('Overfitting Monitor', fontsize=12, fontweight='bold')
    axes[1, 0].legend(frameon=True, shadow=True)
    axes[1, 0].grid(True, alpha=0.3)
    
    if 'learning_rate' in history:
        axes[1, 1].plot(epochs, history['learning_rate'], 'o-', linewidth=2, 
                       markersize=4, color='purple', alpha=0.8)
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=11)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_performance_metrics(y_true, y_pred, save_path='output/plots/performance.png'):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    valid_subjects = support > 0
    subjects = np.where(valid_subjects)[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold', y=0.995)
    
    axes[0, 0].hist(f1[valid_subjects], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(f1[valid_subjects].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {f1[valid_subjects].mean():.3f}')
    axes[0, 0].set_xlabel('F1-Score', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('F1-Score Distribution Across Subjects', fontsize=12, fontweight='bold')
    axes[0, 0].legend(frameon=True, shadow=True)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].scatter(recall[valid_subjects], precision[valid_subjects], 
                      s=support[valid_subjects]*10, alpha=0.6, c=f1[valid_subjects],
                      cmap='viridis', edgecolors='black', linewidth=0.5)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0, 1].set_xlabel('Recall', fontsize=11)
    axes[0, 1].set_ylabel('Precision', fontsize=11)
    axes[0, 1].set_title('Precision vs Recall', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 1.05])
    axes[0, 1].set_ylim([0, 1.05])
    
    top_indices = np.argsort(f1[valid_subjects])[-10:]
    top_subjects = subjects[top_indices]
    top_f1 = f1[top_subjects]
    
    axes[1, 0].barh(range(10), top_f1, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_yticks(range(10))
    axes[1, 0].set_yticklabels([f'S{s+1:03d}' for s in top_subjects])
    axes[1, 0].set_xlabel('F1-Score', fontsize=11)
    axes[1, 0].set_title('Top 10 Subjects', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].set_xlim([0, 1.05])
    
    bottom_indices = np.argsort(f1[valid_subjects])[:10]
    bottom_subjects = subjects[bottom_indices]
    bottom_f1 = f1[bottom_subjects]
    
    axes[1, 1].barh(range(10), bottom_f1, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1, 1].set_yticks(range(10))
    axes[1, 1].set_yticklabels([f'S{s+1:03d}' for s in bottom_subjects])
    axes[1, 1].set_xlabel('F1-Score', fontsize=11)
    axes[1, 1].set_title('Bottom 10 Subjects', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].set_xlim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def generate_report(y_true, y_pred, history, save_path='output/report.txt'):
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    valid_subjects = support > 0
    num_errors = np.sum(y_true != y_pred)
    
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    best_val_acc = max(history['val_acc'])
    
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    loss_gap = final_val_loss - final_train_loss
    overfitting_status = "GOOD" if loss_gap < 0.5 else "MODERATE" if loss_gap < 1.0 else "HIGH"
    
    report = f"""Results Summary

Dataset: PhysioNet EEG Motor Movement/Imagery (109 subjects)
Validation: {len(y_true)} samples from {np.sum(valid_subjects)} subjects

Architecture
  CNN-LSTM hybrid | 2 conv + 2 LSTM layers
  Dropout: {0.2} (spatial) + {0.4} (FC)
  Regularization: L2 (5e-5)
  Features: Time-frequency spectrograms

Training
  Epochs: {best_epoch}/{len(history['train_loss'])} (best)
  Val loss: {best_val_loss:.4f} | Val acc: {best_val_acc:.2f}%
  Early stop: {'Yes' if len(history['train_loss']) < 75 else 'No'}

Overfitting Check
  Train loss: {final_train_loss:.4f} | Val loss: {final_val_loss:.4f}
  Gap: {loss_gap:.4f} [{overfitting_status}]

Performance
  Accuracy: {accuracy*100:.2f}%
  F1-Score: {f1_macro:.3f} (macro) | {f1_weighted:.3f} (weighted)
  Errors: {num_errors}/{len(y_true)} ({(num_errors/len(y_true))*100:.1f}%)

Per-Subject Stats
  F1 mean: {f1[valid_subjects].mean():.3f} | std: {f1[valid_subjects].std():.3f}
  Range: [{f1[valid_subjects].min():.3f}, {f1[valid_subjects].max():.3f}]
  Perfect: {np.sum(f1[valid_subjects] == 1.0)}/{np.sum(valid_subjects)}

Notes
  - {'Model generalizes well' if loss_gap < 0.5 else 'Consider more regularization' if loss_gap < 1.0 else 'Overfitting detected'}
  - {'Consistent across subjects' if f1[valid_subjects].std() < 0.2 else 'Variable subject performance'}
  - Test cross-session robustness recommended
"""
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    return report
