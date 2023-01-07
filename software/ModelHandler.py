import torch
from scipy.signal import butter, lfilter, filtfilt

from scipy import signal
import numpy as np
import pywt

from .models import *

CUTOFF_FREQ = 100
SAMPLING_RATE = 5500
INPUT_SIZE = 27500

def STFT(x, f_s=SAMPLING_RATE, nperseg=4000):
    f, t, Zxx_2_0 = signal.stft(np.abs(x), f_s, nperseg=nperseg)
    return np.abs(Zxx_2_0)

def downsample(x, interval=250):
    num_iter = int(x.shape[0]/interval)
    downsampled = np.zeros(num_iter)
    for i in range(num_iter):
        downsampled[i] = x[i*interval]
    return downsampled

def DWT(signal, f_s=SAMPLING_RATE, interval_ms=200):
    """
      https://sigmobile.org/mobicom/2015/papers/p65-wangA.pdf, 
      https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
    """
    samples_200_ms = int(f_s*interval_ms/1000)
    num_iter = int(signal.shape[0]/samples_200_ms)
    energies_over_time = np.zeros((12, num_iter))
    for i in range(num_iter):
        signal_snippet = signal[i*samples_200_ms:i *
                                samples_200_ms + samples_200_ms]
        detail_components = pywt.wavedec(signal_snippet, 'haar', level=12)[1:]
        energies = np.array([np.sum(np.power(components, 2))
                            for components in detail_components])
        energies_over_time[:, i] = energies
    return energies_over_time

def butterworth(signal, cutoff_freq=CUTOFF_FREQ, sampling_rate=SAMPLING_RATE, order=4):
    """Butterworth low-pass filter"""
    nyquist = 1/2 * sampling_rate
    normal_cutoff_freq = cutoff_freq/nyquist
    b, a = butter(order, normal_cutoff_freq, btype="low", analog=False)
    signal_filtered = lfilter(b, a, signal)
    return signal_filtered

def normalize_min_max(signal: np.ndarray) -> np.ndarray:
    xmin, xmax = np.amin(signal), np.amax(signal)
    return (signal - xmin ) / (xmax-xmin)

class ModelHandler:
    def __init__(self, model_path):
        # Load model dict
        self.model_dict = torch.load(model_path)

        # Initialize the model
        self.base_model = self.model_dict['base_model']
        self.model = model_types[self.base_model]()

        # Loads in the weights of the model
        self.model.load_state_dict(self.model_dict['model_state_dict'])

        # Mean and standardeviation.
        #self.mean = self.model_dict['data_mean']
        #self.std = self.model_dict['data_std']
        if self.base_model == "2DCNN":
            self.dwt_mean = self.model_dict['dwt_mean']
            self.dwt_std = self.model_dict['dwt_std']

        # Datasets and training parameters
        self.datasets = self.model_dict['datasets']
        self.epochs = self.model_dict['epoch']

        # Metrics
        self.train_loss = self.model_dict['train_loss']
        self.train_accuracy = self.model_dict['train_accuracy']
        self.test_accuracy = self.model_dict['test_accuracy']
        self.f1_score = self.model_dict['f1_score']
        self.num_train_samples = self.model_dict['num_train_samples']
        self.num_test_samples = self.model_dict['num_test_samples']
        self.batch_size = self.model_dict['batch_size']
    
    def get_relevant_metrics(self):
        irrelevant_metrics = {"optimizer_state_dict", "model_state_dict"}
        return {key:val for key, val in self.model_dict.items() if key not in irrelevant_metrics}

    def predict(self, data):
        '''
        Predicts the outcome of the input data. It will give the predicted class and also the certainty.
        '''
        # Absolute value of data
        data = np.abs(data)

        # Normalize and pre-process
        data = normalize_min_max(data)
        data = butterworth(data)
        if self.base_model == "1DCNN":
            data = downsample(data)
        elif self.base_model == "2DCNN":
            data = DWT(data)
            data = (data - self.dwt_mean)/self.dwt_std
            
        sample = torch.tensor(data, dtype=torch.float)
        # second unsqueeze to add batch dimension (needed for model).
        sample = sample.unsqueeze(dim=0).unsqueeze(dim=0)
        # Assertion to check if dimension is supported.
        assert sample.dim() <= 4, \
            'The input data has too many dimensions, max is 2 but the given input has {}.'.format(
                sample.dim() - 2)
        sample = sample.permute((1, 0, 2)) if sample.dim(
        ) == 3 else sample.permute((1, 0, 2, 3))

        self.model.eval()
        prob = self.model(sample)

        return torch.argmax(prob).detach().numpy(), np.ravel(prob.detach().numpy())

