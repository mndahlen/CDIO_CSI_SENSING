import torch
import torch.nn as nn
import torch.nn.functional as F

class OneDCNN(nn.Module):
    def __init__(self):
        super(OneDCNN, self).__init__()

        INPUT_SIZE = 110
        N = INPUT_SIZE
        n_filter = 125
        kernel_size = 30
        stride = 1
        self.n_out = int((N - kernel_size)/stride + 1)

        self.conv1 = nn.Conv1d(1, n_filter, kernel_size)
        self.fc1 = nn.Linear(n_filter, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.max_pool1d(x, self.n_out)
        x = torch.transpose(x, 1, 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=-1).squeeze(1)
        return x

class TwoDCNN(nn.Module):    
    def __init__(self):
        super(TwoDCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

model_types = {
    '1DCNN': OneDCNN,
    '2DCNN': TwoDCNN
}
