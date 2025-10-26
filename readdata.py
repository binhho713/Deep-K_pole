import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import scipy.stats.mstats as statm
import pandas as pd
import random
import logging
import time
import itertools
import math
from torch.autograd import Variable
import torch.nn as nn
import torch

def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger with a file and console handler."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger

def generate_ts(size, length):

    x = np.zeros((size, length))

    for i in range(size):
        x[i] = np.random.normal(loc=0.0, scale=1.0, size=length)

    np.save('base_ts_%s.npy' % size, x)
    s = np.zeros((size * 1000, length))
    j = 0
    for i in range(len(s)):
        h = np.random.normal(loc=0.0, scale=1.0, size=length)
        for k in range(length):
            t = x[i]*h[k]
            noise = np.random.normal(loc=0.0, scale=0.04, size=length)
            t += noise
            l = i
            s[j] = np.concatenate((t, np.array([l])), axis=0)
            j += 1

    np.save('train_ts_100.npy', s)

#generate_ts()

def generate_train_ts(x, nametag):

    size = x.shape[0]
    length = x.shape[1]
    s = np.zeros((size * 1000, length + 1))
    j = 0
    for i in range(size):
        h = np.random.normal(loc=0.0, scale=1.0, size=1000)
        for k in range(1000):
            t = x[i]*h[k]
            noise = np.random.normal(loc=0.0, scale=0.04, size=length)
            t += noise
            l = i
            s[j] = np.concatenate((t, np.array([l])), axis=0)
            j += 1

    # np.save('train_ts_%d_%s.npy' % (size, nametag), s)
    np.save(nametag, s)


def generate_train_ts_v2(x, nametag, samples_per_series=1000, noise_scale=0.04, s_or_r=True):
    # Validate input
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError("Input `x` must be a 2D NumPy array (n_series, series_length).")

    size, length = x.shape
    total_samples = size * samples_per_series

    # Generate random scaling factors and noise for all samples at once
    h = np.random.normal(loc=0.0, scale=1.0, size=(size, samples_per_series))
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=(size, samples_per_series, length))

    # Broadcast to create synthetic samples
    synthetic_data = x[:, None, :] * h[..., None] + noise

    # Reshape to (total_samples, series_length)
    synthetic_data = synthetic_data.reshape(total_samples, length)

    # Add labels
    labels = np.repeat(np.arange(size), samples_per_series)
    synthetic_data_with_labels = np.hstack([synthetic_data, labels[:, None]])

    # Save the data
    if s_or_r:
        np.save(nametag, synthetic_data_with_labels)
    else:
        return synthetic_data_with_labels

def generate_train_ts_with_labels(input, labels, nametag, sample, l_opt, s_or_r):

    x = statm.zscore(input, axis=0)
    size = x.shape[0]
    length = x.shape[1]
    if l_opt:
        s = np.zeros((size * sample, length + 4))
    else:
        s = np.zeros((size * sample, length))
    j = 0
    for i in range(size):
        h = np.random.normal(loc=0.0, scale=1.0, size=sample)
        for k in range(sample):
            t = x[i]*h[k]
            noise = np.random.normal(loc=0.0, scale=0.04, size=length)
            t += noise
            t = statm.zscore(t)
            if l_opt:
                l = labels[i]
                s[j] = np.concatenate((t, np.array([l])[0]), axis=0)
            else:
                s[j] = t
            j += 1

    # np.save('train_ts_%d_%s.npy' % (size, nametag), s)
    if s_or_r:
        np.save(nametag, s)
    else:
        return s

def generate_mixture(x, k):
    mixture = []
    labels = []
    indice = np.arange(0, len(x), 1)
    arr = np.zeros((k, x.shape[1]))

    for combo in itertools.combinations(indice, k):
        for i in range(k):
            arr[i] = x[int(combo[i])]
        z = np.random.choice([-1, 1], size=k)
        coef = np.random.rand(k)
        coef *= z
        coef /= sum(coef)
        arr *= coef[:, np.newaxis]
        mixture.append(sum(arr))
        labels.append(combo)

    return mixture, labels

def generate_mix_ts(x, nametag, sample, train_ratio, l_opt, s_or_r):
    base_mix_ts = []
    base_labels = []
    mix_ts = []
    labels = []
    test_ts = []
    test_labels = []

    #generate base mixture
    for i in range(len(x)):
        for j in range(len(x)):
            if j > i:
                z = np.random.choice([-1, 1], size=2)
                k1 = np.random.rand()
                k2 = (1 - k1)
                base_mix_ts.append(x[i] * k1 * z[0] + x[j] * k2 * z[1])
                base_labels.append([i, j])

    y = dict(mix_ts=base_mix_ts, labels=base_labels)

    if s_or_r:
        scipy.io.savemat('base_mix_ts_%s.mat' % nametag, y)

    #generate mixture samples
    for i in range(len(base_mix_ts)):
        h = np.random.normal(loc=0.0, scale=1.0, size=sample)
        for j in range(sample):
            t = base_mix_ts[i] * h[j]
            noise = np.random.normal(loc=0.0, scale=0.04, size=(x.shape[1]))
            t += noise
            if j < sample * train_ratio:
                mix_ts.append(t)
                if l_opt:
                    labels.append(base_labels[i])
            else:
                test_ts.append(t)
                if l_opt:
                    test_labels.append(base_labels[i])

    s = dict(mix_ts=mix_ts, labels=labels)
    t = dict(mix_ts=test_ts, labels=test_labels)
    if s_or_r:
        scipy.io.savemat('mix_ts_10_%s.mat' % nametag, s)
        scipy.io.savemat('test_ts_10_%s.mat' % nametag, t)
    else:
        return t, s

def gaussian(peak, mu, standard_deviation):
    # peak 200, 1
    # mu 200, 1
    # std 1
    variance = standard_deviation**2
    x = np.arange(0, 1024, 1)
    output = np.exp(-((x - mu)**2)/(2.0 * variance)) * peak #200, 1024
    output = np.sum(output, axis=0)

    return output

def synthetic_time_series(size, length, meaningful_ratio, normalize=False):
    """
    Generate synthetic time series data with realistic patterns.

    Parameters:
    - size (int): Total number of time series.
    - length (int): Length of each time series.
    - meaningful_ratio (float): Fraction of series that are meaningful (0 to 1).
    - normalize (bool): Whether to normalize the data using Z-score normalization.

    Returns:
    - np.ndarray: Array of generated time series with meaningful series first.
    """
    n_series = size
    series_length = length
    n_meaningful = int(size * meaningful_ratio)  # Convert to integer
    n_white_noise = n_series - n_meaningful  # Remaining are white noise

    # Generate meaningful time series
    meaningful = []
    for _ in range(n_meaningful):
        pattern_type = np.random.choice(["sine", "random_walk", "trend"])
        t = np.linspace(0, 2 * np.pi, series_length)

        if pattern_type == "sine":
            # Add noise to sine wave
            amplitude = np.random.uniform(0.5, 2)
            frequency = np.random.uniform(0.5, 2)
            noise = np.random.normal(loc=0, scale=0.1, size=series_length)
            series = amplitude * np.sin(frequency * t) + noise

        elif pattern_type == "random_walk":
            # Add slight drift to random walk
            drift = np.random.uniform(-0.1, 0.1)
            random_walk = np.cumsum(np.random.normal(loc=0, scale=1, size=series_length)) + drift * t
            noise = np.random.normal(loc=0, scale=0.1, size=series_length)
            series = random_walk + noise

        elif pattern_type == "trend":
            # Generate exponential or linear trend with noise
            trend_type = np.random.choice(["linear", "exponential"])
            if trend_type == "linear":
                slope = np.random.uniform(-0.1, 0.1)
                intercept = np.random.uniform(-1, 1)
                trend = slope * t + intercept
            else:  # exponential
                base = np.random.uniform(1.01, 1.5)
                trend = base ** t
            noise = np.random.normal(loc=0, scale=0.1, size=series_length)
            series = trend + noise

        meaningful.append(series)

    # Generate white noise time series
    white_noise = [np.random.normal(loc=0, scale=1, size=series_length) for _ in range(n_white_noise)]

    # Combine with meaningful series first
    all_series = meaningful + white_noise

    # Normalize if specified
    all_series = np.array(all_series)
    if normalize:
        all_series = statm.zscore(all_series, axis=1)

    return all_series

def Brute_force(dataPath, k, threshold, run_time):
    duration = run_time * 60 * 60
    start_time = time.time()
    # Load data with memory mapping
    data = np.load(dataPath, mmap_mode='r')

    # Create a range of indices for combinations
    cb = range(len(data))
    mlp = []
    strength = []
    arr = np.zeros((k, 325))
    st = True
    print('Start mining multipoles in ' + dataPath)
    # Iterate over all combinations of k indices
    for i in itertools.combinations(cb, int(k)):
        for j in range(k):
            arr[j] = data[i[j]][:325]

        arr = statm.zscore(arr, axis=1)
        x = np.transpose(arr)
        x = np.corrcoef(x, rowvar=False)
        x = np.nan_to_num(x)

        eigenvalues, eigenvectors = np.linalg.eig(x)
        min_variance_index = np.argmin(eigenvalues)
        min_variance_eigenvector = eigenvectors[:, min_variance_index]

        s = 0
        for j in range(k):
            s += arr[j] * min_variance_eigenvector[j]

        if np.var(s) <= threshold:
            mlp.append(i)
            strength.append(np.var(s))

        # Check if the specified run time has elapsed
        current_time = time.time()
        if (current_time - start_time) >= duration:
            st = False
            print('Breaking loop due to time constraint.')
            break
    # Save the results to a .mat file
    print('run for %d' % (time.time() - start_time))
    output = dict(mlp=mlp, strength= strength, finished=st, runTime=(time.time() - start_time))
    scipy.io.savemat('brute_force_%d.mat' % len(data), output)
    print('Finished')

def Brute_force_v2(data, nametag, k, threshold, linear_gain=False):
    mlp = []
    st = []
    eg = []
    cv = []
    lg = []

    arr = np.zeros((k, data.shape[1]))

    start_time = time.time()

    for combo in itertools.combinations(range(len(data)), k):
        arr[:k] = [data[int(x)] for x in combo]

        arr = statm.zscore(arr, axis=1)
        x = np.corrcoef(arr.T, rowvar=False)
        x = np.nan_to_num(x)
        cv.append(x)

        eigenvalues, eigenvectors = np.linalg.eig(x)
        min_variance_index = np.argmin(eigenvalues)
        min_variance_eigenvector = eigenvectors[:, min_variance_index]
        eg.append(min_variance_eigenvector)

        out = np.var(np.dot(arr.T, min_variance_eigenvector))

        mlp.append(combo)
        st.append(out)

        if linear_gain:
            ll = []
            for cb in itertools.combinations(combo, k-1):
                sub_arr = np.array([data[int(x)] for x in cb])
                sub_arr = statm.zscore(sub_arr, axis=1)
                sub_x = np.corrcoef(sub_arr.T, rowvar=False)
                sub_x = np.nan_to_num(sub_x)

                sub_eigenvalues, sub_eigenvectors = np.linalg.eig(sub_x)
                sub_min_vector = sub_eigenvectors[:, np.argmin(sub_eigenvalues)]

                linear_gain = np.var(np.dot(sub_arr.T, sub_min_vector)) - out
                ll.append(linear_gain)

            lg.append(ll)

    if linear_gain:
        df = pd.DataFrame({
            'Multipoles': mlp,
            'Strength': st,
            'Linear Gain': lg,
            'Eigenvector': eg,
            'Covariance': cv,
        })
    else:
        # Fill `lg` with placeholders to match the length of other columns
        lg = [None] * len(mlp)
        df = pd.DataFrame({
            'Multipoles': mlp,
            'Strength': st,
            'Linear Gain': lg,  # Placeholder values
            'Eigenvector': eg,
            'Covariance': cv,
        })

    df.to_csv(f'{nametag}.csv', index=False)
    print(f'Finished after {time.time() - start_time:.2f} seconds')

def linear_dependence (input):

    arr = statm.zscore(input, axis=1)
    x = np.corrcoef(arr.T, rowvar=False)
    x = np.nan_to_num(x)

    eigenvalues, eigenvectors = np.linalg.eig(x)
    min_variance_index = np.argmin(eigenvalues)
    min_variance_eigenvector = eigenvectors[:, min_variance_index]

    linear_dependence = 1 - np.var(np.dot(arr.T, min_variance_eigenvector))

    return linear_dependence

def k_pole_evaluator (input, predict, top=1):

    result = []

    for i in range(top):
        mp = np.concatenate(input, predict)
        result.append(linear_dependence(mp))

    return linear_dependence

batch_size = 64
latent_dim = 100
n_classes = 150
ts_size = 325
nums = 150
use_cuda = torch.cuda.is_available()
is_training = 1

class Generator(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=200, output_dim=325, embedding_dim=200):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(condition_dim, embedding_dim)
        self.ln1 = nn.Linear(noise_dim, 256)
        self.ln2 = nn.Linear(embedding_dim, 256)
        self.model = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 768),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(768, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_dim)
        )

    def forward(self, noise, condition):
        y = self.embedding(condition)
        y = self.ln2(y)
        x = self.ln1(noise)
        x = torch.cat([x, y], dim=1)
        return self.model(x)


import torch.nn.functional as F

seq_length = 325
num_classes = 200
class Generator1(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=171, output_dim=108, embedding_dim=100):
        super(Generator1, self).__init__()
        self.embedding = nn.Embedding(condition_dim, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + embedding_dim, 216),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(216, 648),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(648, 864),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(864, 1080),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1080, output_dim)
        )

    def forward(self, noise, condition):
        y = self.embedding(condition)
        x = torch.cat([noise, y], dim=1)
        return self.model(x)

class Generator2(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=171, output_dim=108, embedding_dim=171):
        super(Generator2, self).__init__()
        self.embedding = nn.Embedding(condition_dim, embedding_dim)
        self.ln1 = nn.Linear(noise_dim, 256)
        self.ln2 = nn.Linear(embedding_dim, 256)
        self.model = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_dim)
        )

    def forward(self, noise, condition):
        y = self.embedding(condition)
        y = self.ln2(y)
        x = self.ln1(noise)
        x = torch.cat([x, y], dim=1)
        return self.model(x)

class Generator3(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=171, output_dim=108, embedding_dim=171):
        super(Generator3, self).__init__()
        self.embedding = nn.Embedding(condition_dim, embedding_dim)
        self.ln1 = nn.Linear(noise_dim, 512)
        self.ln2 = nn.Linear(embedding_dim, 512)
        self.model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, output_dim)
        )

    def forward(self, noise, condition):
        y = self.embedding(condition)
        y = self.ln2(y)
        x = self.ln1(noise)
        x = torch.cat([x, y], dim=1)
        return self.model(x)

class Generator4(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=200, output_dim=325, embedding_dim=200):
        super(Generator4, self).__init__()
        self.embedding = nn.Embedding(condition_dim, embedding_dim)
        self.ln1 = nn.Linear(noise_dim, 512)
        self.ln2 = nn.Linear(embedding_dim, 512)
        self.model = nn.Sequential(
            nn.Linear(1024, 1280),
            nn.ReLU(),
            nn.Linear(1280, 1536),
            nn.ReLU(),
            nn.Linear(1536 , 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim)
        )

    def forward(self, noise, condition):
        y = self.embedding(condition)
        y = self.ln2(y)
        x = self.ln1(noise)
        x = torch.cat([x, y], dim=1)
        return self.model(x)

class Generator5(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=171, output_dim=108, embedding_dim=171):
        super(Generator5, self).__init__()
        self.embedding = nn.Embedding(condition_dim, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1536),
            nn.ReLU(),
            nn.Linear(1536, output_dim)
        )

    def forward(self, noise, condition):
        y = self.embedding(condition)
        x = torch.cat([noise, y], dim=1)
        return self.model(x)

class Generator6(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=150, output_dim=325, embedding_dim=150):
        super(Generator6, self).__init__()
        self.embedding = nn.Embedding(condition_dim, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + embedding_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1536),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1536, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 3072),
            nn.ReLU(),
            nn.Linear(3072, output_dim),
        )

    def forward(self, noise, condition):
        y = self.embedding(condition)
        x = torch.cat([noise, y], dim=1)
        return self.model(x)


class Generator9(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=171, output_dim=108, embedding_dim=171):
        super(Generator9, self).__init__()
        self.embedding = nn.Embedding(condition_dim, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1536),
            nn.ReLU(),
            nn.Linear(1536, output_dim),
        )

    def forward(self, noise, condition):
        y = self.embedding(condition)
        x = torch.cat([noise, y], dim=1)
        return self.model(x)


generator = Generator9()
if torch.cuda.is_available():
    print("use cuda")
    generator = generator.cuda()
generator.load_state_dict(torch.load("C:/Users/hotha/Downloads/FEDX_data/G_ts_client_4-180.model", weights_only=True))
generator.eval()

data = scipy.io.loadmat('C:/Users/hotha/Downloads/Dpoles-ex-brute/CLIQ_COMPLETE_2_Multipoles_ASV0.8_mu_0_sigma_0.1_delta_0.15')
mlp = data['FinalMPList'][0]
st = data['FinalLEVList'][0]
check = []
ld = []
for i, m in enumerate(mlp):
    if 118 in m:
        check.append(m)
        ld.append(st[i])

out = 1 - np.min(ld)

df = pd.DataFrame({
        'Multipoles': mlp,
        'Strength': st,
    })
df.to_csv(f'comet_ASV2_output.csv', index=False)
print('hahaha')
# mixture, labels = generate_mixture(subset, 4)
#
# range_values = list(range(0, len(labels)))
# random_values = random.sample(range_values, 150)
# x = np.zeros((150, subset.shape[1]))
# lab = np.zeros((150, 4))
# for i in range(150):
#     x[i] = mixture[int(random_values[i])]
#     lab[i] = labels[int(random_values[i])]
#
# generate_train_ts_with_labels(x, lab, nametag='test_mixture_150_psl.npy', sample=1000, l_opt=True, s_or_r=True)
t2 = np.load('base_ts_171_psl.npy')
# generate_train_ts_v2(t2, 'train_cGAN_psl.npy', samples_per_series=2000, noise_scale=0.05)
# ababa = np.load('train_cGAN_ASV.npy')
# t, _ = np.split(t2, (150,), axis=0)
# generate_train_ts_v2(t, 'train_cGAN_ASV.npy', samples_per_series=1000, noise_scale=0.05)
# mixture, labels = generate_mixture(t, 3)
# range_values = list(range(0, len(labels)))
# random_values = random.sample(range_values, 40)
# x = np.zeros((40, 325))
# lab = np.zeros((40, 3))
# for i in range(40):
#     x[i] = mixture[int(random_values[i])]
#     lab[i] = labels[int(random_values[i])]
# generate_train_ts_with_labels(x, lab, nametag='test_mixture_40_ASV.npy', sample=1000, l_opt=True, s_or_r=True)

# vv = np.var(t2, axis=1)
#

# condition_np = np.full(batch_size, 10, dtype=np.int64)
# condition = torch.tensor(condition_np, dtype=torch.long).cuda()
# noise = torch.randn(batch_size, 100).cuda()
# ts = generator(noise, condition)
#
# plt.plot(t2[2])
# plt.plot(ts[k + 0])
# plt.plot(ts[k + 1])
# plt.plot(ts[k + 2])
# plt.plot(ts[k + 3])
# plt.plot(ts[k + 4])
# plt.plot(ts[k + 5])
# plt.plot(ts[k + 6])
# plt.plot(ts[k + 7])
# plt.plot(ts[k + 8])
# plt.plot(ts[k + 9])
# plt.plot(ts[k + 10])
# plt.show()
#
# import encoder_ASV
# pred = encoder_ASV.resnet18(predictor=True)
# pred.load_state_dict(torch.load("C:/Users/hotha/Downloads/pred_DMD_ASV_2_60.model"))
# pred.eval().cuda()
# # sep = encoder_psl.resnet18(predictor=False)
# # sep.load_state_dict(torch.load("C:/Users/hotha/Downloads/sep_DMD_psl_2220.model"))
# # sep.eval().cuda()
#
# choosen_one = 122
# size = 108
#
# test_int = torch.tensor(t2[choosen_one].reshape(-1, 1, size)).float().cuda()
# # emb = sep(test_int)
# # wtf = emb.cpu().data.numpy()
# pred_dis = pred(test_int)
# # gen_labels_0 = torch.LongTensor(np.arange(n_classes, dtype = "int32")).cuda()
# # gen_ts_0 = generator(emb.view(-1,100), gen_labels_0)
# # suffer = gen_ts_0.cpu().data.numpy()
# # gen_ts = gen_ts_0.permute(1, 0) * pred_dis.view(-1)#108, 10
# # gen_ts = gen_ts.permute(1, 0).view(n_classes, 1, ts_size) # 171, 108
# # sorrow = gen_ts.cpu().data.numpy()
# # gen_ts = torch.sum(gen_ts, dim= 0) #bs, 1, 1024
# # pain = gen_ts.cpu().data.numpy()
#
# mlp = []
# st = []
# eg = []
# cv = []
# lg = []
#
# arr = np.zeros((4, size))
# arr[0] = t2[choosen_one]
# for combo in itertools.combinations(indice, 3):
#     arr[-3:] = [t2[int(x)] for x in combo]
#
#     arr = statm.zscore(arr, axis=1)
#     x = arr.copy()
#     x = np.transpose(x)
#     x = np.corrcoef(x, rowvar=0)
#     x = np.nan_to_num(x)
#     cv.append(x)
#     eigenvalues, eigenvectors = np.linalg.eig(x)
#     min_variance_index = np.argmin(eigenvalues)
#     min_variance_eigenvector = eigenvectors[:, min_variance_index]
#     eg.append(min_variance_eigenvector)
#     out = 0
#     for j in range(len(arr)):
#         out += arr[j] * min_variance_eigenvector[j]
#     out = np.var(out)
#     ml = np.zeros((4), dtype=int)
#     ml[0] = choosen_one
#     ml[-3:] = [x for x in combo]
#
#     ll = np.zeros((4))
#     arr1 = np.zeros((3, size))
#
#     for index, cb in enumerate(itertools.combinations(ml, 3)):
#         arr1[-3:] = [t2[int(x)] for x in cb]
#
#         arr1 = statm.zscore(arr1, axis=1)
#         x = arr1.copy()
#         x = np.transpose(x)
#         x = np.corrcoef(x, rowvar=0)
#         x = np.nan_to_num(x)
#         eigenvalues, eigenvectors = np.linalg.eig(x)
#         min_variance_index = np.argmin(eigenvalues)
#         min_variance_eigenvector = eigenvectors[:, min_variance_index]
#         str = 0
#         for j in range(len(arr1)):
#             str += arr1[j] * min_variance_eigenvector[j]
#         str = np.var(str)
#         ll[index] = str - out
#
#     mlp.append(ml)
#     st.append(out)
#     lg.append(ll)
#
# df = pd.DataFrame({
#     'Multipoles': mlp,
#     'Strength': st,
#     'Linear GAin': lg,
#     'k': eg,
#     'covariance': cv,
# })
# df.to_csv(f'C:/Users/hotha/Downloads/eval_DMD_ASV_23_11_ts_{choosen_one}.csv', index=False)

# plt.plot(t2[9])
# plt.plot(pain[0])
# plt.plot(statm.zscore(sorrow[9][0]), "green")
# plt.plot(suffer[0])
# plt.plot(suffer[1])
# plt.plot(suffer[2])
# plt.plot(suffer[3])
# plt.plot(suffer[4])
# plt.plot(suffer[5])
# plt.plot(suffer[6])
# plt.plot(suffer[7])
# plt.plot(suffer[8])
# plt.plot(suffer[9])
# plt.plot(suffer[10])
# plt.plot(t2[2], "g")
# plt.plot(pain[0], "r")
# plt.plot(sorrow[0][0])
# plt.plot(sorrow[1][0])
# plt.plot(sorrow[2][0])
# plt.plot(sorrow[3][0])
# plt.plot(sorrow[4][0])
# plt.plot(sorrow[6][0])
# plt.plot(sorrow[7][0])
# plt.plot(sorrow[8][0])
#plt.plot(statm.zscore(sorrow[17][0]), "blue")
#plt.plot(statm.zscore(sorrow[34][0]), "purple") #sai
#plt.plot(statm.zscore(sorrow[148][0]), "black")
#plt.plot(statm.zscore(t2[17]), "r")
#plt.plot(statm.zscore(t2[34]), "r")
#plt.plot(statm.zscore(t2[148]), "r") #hoi dung van sai
#plt.show()

torch.manual_seed(0)
z = torch.randn(batch_size, latent_dim).cuda()
wth = z.cpu().data.numpy()
gen_labels = Variable(torch.cuda.LongTensor(np.random.randint(0, n_classes, batch_size)))
gen_labels1 = Variable(torch.cuda.LongTensor(np.zeros(batch_size)))
gen_labels1 = Variable(torch.full(size=(batch_size,), fill_value=10, dtype=torch.long).cuda())
gen_ts = generator(z, gen_labels1)
hello_bitch = gen_ts.cpu().data.numpy()
la = gen_labels1.cpu().data.numpy()

k = 0
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(t2[10])
# axs[0, 0].plot(statm.zscore(hello_bitch[k+10]))
# axs[0, 0].plot(statm.zscore(hello_bitch[k+20]))
# axs[0, 0].plot(statm.zscore(hello_bitch[k+40]))
axs[0, 0].plot(hello_bitch[k+10])
axs[0, 0].plot(hello_bitch[k+20])
axs[0, 0].plot(hello_bitch[k+40])
# axs[0, 0].plot(hello_bitch[k+10])
# axs[0, 0].plot(hello_bitch[k+20])
# axs[0, 0].plot(hello_bitch[k+40])
axs[0, 0].set_title("OG time series")
axs[0, 1].plot(hello_bitch[k])
axs[0, 1].plot(hello_bitch[k+2])
axs[0, 1].plot(hello_bitch[k+4])
axs[0, 1].plot(hello_bitch[k+6])
axs[0, 1].plot(hello_bitch[k+8])
axs[0, 1].plot(hello_bitch[k+12])
axs[0, 1].plot(hello_bitch[k+14])
axs[0, 1].plot(hello_bitch[k+16])
axs[0, 1].plot(hello_bitch[k+18])
axs[0, 1].plot(hello_bitch[k+22])
axs[0, 1].plot(hello_bitch[k+24])
axs[0, 1].plot(hello_bitch[k+31])
axs[0, 1].plot(hello_bitch[k+33])
axs[0, 1].plot(hello_bitch[k+35])
axs[0, 1].plot(hello_bitch[k+44])
axs[0, 1].plot(hello_bitch[k+48])
axs[0, 1].plot(hello_bitch[k+57])
axs[0, 1].plot(hello_bitch[k+59])
axs[0, 1].plot(hello_bitch[k+61])
axs[0, 1].plot(hello_bitch[k+63])
axs[0, 1].set_title("linear_cGAN generated")
axs[1, 0].plot(t2[20])
axs[1, 0].plot(t2[3], "g")
plt.show()

# base_psl = np.load("base_ts_171_psl.npy", allow_pickle=True)
# base_psl = statm.zscore(base_psl, axis=1)
#
# base_psl_40 = base_psl[:40]
#
# generate_train_ts(base_psl_40, nametag = 'train_cGAN_psl_40.npy')
#
# mixture, labels = generate_mixture(base_psl_40, 3)
# range_values = list(range(0, len(labels)))
# random_values = random.sample(range_values, 40)
# x = np.zeros((40, 108))
# lab = np.zeros((40, 3))
# for i in range(40):
#     x[i] = mixture[int(random_values[i])]
#     lab[i] = labels[int(random_values[i])]
# generate_train_ts_with_labels(x, lab, nametag='test_mixture_40_psl.npy', sample=1000, l_opt=True, s_or_r=True)

# trainData = np.load('train_cGAN_psl_40.npy',allow_pickle=True)
# time_series, truth_labels_list = np.split(trainData, (trainData.shape[1]-2,), axis=1)
# print(truth_labels_list)




