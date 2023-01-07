from ModelHandler import STFT, downsample, DWT, butterworth, normalize_min_max
from models import OneDCNN, TwoDCNN
import torch
import os
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings("ignore")

model_name = "2DCNN"#1DCNN or 2DCNN
num_epochs = 50
IN = 0
OUT = 1

class SignalDataset(Dataset):
    def __init__(self, data, target):
      super(Dataset, self).__init__()
      self.data = data
      self.target = target
    
    def __len__(self):
      return len(self.data)

    def __getitem__(self, index):
      curr_sample = self.data[index,:]
      curr_label = self.target[index]
      return (
          torch.tensor(curr_sample, dtype=torch.float),
          torch.tensor(curr_label, dtype=torch.int)
      )

    def signal_size(self):
      return self.data.shape[1]

def test(model, testloader, model_name, use_cuda=False):
    model.eval()
    preds_full = np.asarray([])
    targets_full = np.asarray([])
    for batch in testloader:
        inputs, targets = batch

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        inputs, targets = torch.tensor(inputs, requires_grad=False), torch.tensor(targets).long()
        inputs=inputs.unsqueeze(dim=0)
        if model_name=="1DCNN":
            inputs=inputs.permute((1,0,2))
        elif model_name=="2DCNN":
            inputs=inputs.permute((1,0,2,3))

        outputs = model(inputs)

        preds_full = np.concatenate((preds_full, torch.argmax(outputs, axis=1).numpy()), axis=0)
        targets_full = np.concatenate((targets_full, targets.numpy()), axis=0)

    accuracy = accuracy_score(preds_full, targets_full)
    f1score = f1_score(preds_full, targets_full)

    return (accuracy, f1score)

# load data
christian_1 = "data/Christian_Skrubben"
christian_2 = "data/chrigu_alternating_1_setup_5_221104"
christian_3 = "data/chrigu_alternating_2_setup_5_221104"
christian_4 = "data/chrigu_alternating_3_setup_5_221103"
joel_1 = "data/Joel_Skrubben"
joel_2 = "data/joehe_alternating_1_setup_5_221103"
joel_3 = "data/joehe_alternating_2_setup_5_221104"
henrik_1 = "data/Henrik_Skrubben"
martin_1 = "data/Martin_Skrubben"
sebastian_1 = "data/Sebastian_Skrubben"
tiger_1 = "data/Tiger_Skrubben"
dirs = [christian_1, christian_2, christian_3, christian_4, joel_1, joel_2, joel_3, henrik_1, martin_1, sebastian_1, tiger_1]

X_in = []
X_in_label = []
X_out = []
X_out_label = []

for dir in dirs:
  print(dir)
  file_ = open(os.path.join(dir,"description.json"))
  description = json.load(file_)
  file_.close()
  in_files = description["schemas"]["in"]["files"]
  out_files = description["schemas"]["out"]["files"]

  for filename in in_files:
    X_in.append(np.fromfile(os.path.join(dir,filename + ".iq"), dtype=np.complex64))
    X_in_label.append(IN)

  for filename in out_files:
    X_out.append(np.fromfile(os.path.join(dir,filename + ".iq"), dtype=np.complex64))
    X_out_label.append(OUT)

X_in = np.abs(np.asarray(X_in))
X_out = np.abs(np.asarray(X_out))

# Feature processing
print("Processing features")
X_in = np.apply_along_axis(normalize_min_max, 1, X_in)
X_in = np.apply_along_axis(butterworth, 1, X_in)
X_out = np.apply_along_axis(normalize_min_max, 1, X_out)
X_out = np.apply_along_axis(butterworth, 1, X_out)
if model_name == "1DCNN":
    X_in = np.apply_along_axis(downsample, 1, X_in)
    X_out = np.apply_along_axis(downsample, 1, X_out)
    data, target = np.concatenate((X_in, X_out)), X_in_label + X_out_label
elif model_name == "2DCNN":
    X_in = np.apply_along_axis(DWT, 1, X_in)
    X_out = np.apply_along_axis(DWT, 1, X_out)
    data, target = np.concatenate((X_in, X_out)), X_in_label + X_out_label
    dwt_mean = np.mean(data, axis=0)
    dwt_std = np.std(data,axis=0)
    data = (data - dwt_mean)/(dwt_std + 0.1)
dataset = SignalDataset(data=data, target=target)
print("done")

# Dataloader
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True, num_workers=4)
testloader = DataLoader(dataset=testset, batch_size=4, shuffle=True, num_workers=4)

# Train model
if model_name == "1DCNN":
    model = OneDCNN()
elif model_name == "2DCNN":
    model = TwoDCNN()
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_loss_vec = []
valid_loss_vec = []
preds_full = np.asarray([])
targets_full = np.asarray([])

for epoch in range(0, num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(trainloader):
        inputs, targets = batch
        inputs, targets = torch.tensor(inputs), torch.tensor(targets).long()
        inputs=inputs.unsqueeze(dim=0)
        if model_name == "1DCNN":
            inputs=inputs.permute((1,0,2))
        elif model_name == "2DCNN":
            inputs=inputs.permute((1,0,2,3))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        preds_full = np.concatenate((preds_full, torch.argmax(outputs, axis=1).numpy()), axis=0)
        targets_full = np.concatenate((targets_full, targets.numpy()), axis=0)
        train_accuracy = accuracy_score(preds_full, targets_full)
    print('Loss: {:.8f} | Acc: {:.2f}'.format((train_loss/(batch_idx+1)), train_accuracy))

test_accuracy, f1score = test(model=model, testloader=testloader, model_name=model_name)

print('Test Acc: {:.2f}'.format(test_accuracy))

model_path = 'saved_models/{}_trained_{}'.format(model_name, num_epochs)
if model_name == "1DCNN":
    dwt_mean = None
    dwt_std = None
elif model_name == "2DCNN":
    dwt_mean = dwt_mean
    dwt_std = dwt_std + 0.1

torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'base_model': model_name,
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'f1_score': f1score,
    'datasets': "dataset",
    'num_train_samples': len(trainset),
    'num_test_samples': len(testset),
    'batch_size': 4,
    'dwt_mean':dwt_mean,
    'dwt_std':dwt_std
}, model_path)
