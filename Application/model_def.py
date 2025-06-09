import torch.nn as nn
import torch.nn.functional as F

class CNN_BiLSTM_Model(nn.Module):
    def __init__(self, input_channels=6, num_classes=5):
        super(CNN_BiLSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)

        self.norm = nn.LayerNorm(128)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True, dropout=0.3)

        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.6)  
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
