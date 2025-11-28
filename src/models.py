import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, num_channels, num_freq_bins, num_time_steps, 
                 num_classes=109, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout2d(0.2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout2d(0.2)
        
        conv_out_freq = num_freq_bins // 4
        conv_out_time = num_time_steps // 4
        
        self.lstm = nn.LSTM(
            input_size=64 * conv_out_freq,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.4 if 2 > 1 else 0
        )
        
        self.dropout3 = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time, channels * freq)
        
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        
        x = self.dropout3(x)
        x = self.fc(x)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience
