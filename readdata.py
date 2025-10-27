import scipy.io
import numpy as np
import torch as torch
import scipy.stats.mstats as statm
import pandas as pd
import random
import logging
import time
import itertools
from itertools import combinations

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

def generate_train_ts(data, nametag, labels, s_or_r):
    size = data.shape[0]
    length = data.shape[1]
    s = np.zeros((size * 1000, length + 1))
    j = 0
    l = []
    x = statm.zscore(data, axis=1)

    for i in range(size):
        h = np.random.normal(loc=0.0, scale=1.0, size=1000)
        for k in range(1000):
            t = x[i]*h[k]
            noise = np.random.normal(loc=0.0, scale=0.04, size=length)
            t += noise
            l = i
            s[j] = np.concatenate((t, np.array([l])), axis=0)
            #s[j] = t
            #l.append(labels[i])
            j += 1

    if s_or_r:
        np.save(nametag, s)
    else:
        return s, l

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
	
    if l_opt:
    	s = dict(mix_ts=mix_ts, labels=labels)
    	t = dict(mix_ts=test_ts, labels=test_labels)
    if s_or_r:
        if l_opt:
            scipy.io.savemat(nametag, s)
            scipy.io.savemat('test' + nametag, t)
        else:
            np.save(nametag, s)
            np.save('test' + nametag, t)
    else:
        if l_opt:
            return s, t
        else:
            return mix_ts, labels

def gaussian(peak, mu, standard_deviation):
    # peak 200, 1
    # mu 200, 1
    # std 1
    variance = standard_deviation**2
    x = np.arange(0, 1024, 1)
    output = np.exp(-((x - mu)**2)/(2.0 * variance)) * peak #200, 1024
    output = np.sum(output, axis=0)

    return output

def stick_pattern():

    peak = np.zeros(shape= (10,200,1), dtype='float')
    mu = np.zeros(shape= (10,200,1), dtype='float')
    standard_deviation = np.random.uniform(size=10) + 1e-6
    np.save('standard_deviation.npy', standard_deviation)
    set = np.load('sticks_lib.npy',allow_pickle=True)
    for num in range(10):
        for (i, k) in enumerate(set[num]):
            mu[num][i] = k[0]
            peak[num][i] = k[1]

    stickpattern = dict(mu = mu, peak = peak)
    scipy.io.savemat('stick_patterns.mat', stickpattern)
    ts = []

    for i in range(len(peak)):
        ts.append(gaussian(peak[i], mu[i], standard_deviation[i]))

    np.save('ts_base_GMM', ts)

    s = np.zeros((10 * 1000, 1025))
    j = 0
    for i in range(len(ts)):
        h = np.random.normal(loc=0.0, scale=1.0, size=1000)
        for k in range(1000):
            t = ts[i] * h[k]
            noise = np.random.normal(loc=0.0, scale=0.04, size=1024)
            t += noise
            l = i
            s[j] = np.concatenate((t, np.array([l])), axis=0)
            j += 1

    np.save('train_ts_10_GMM.npy', s)

    generate_mix_ts(ts, 'GMM')

def Brute_force(dataPath, k, threshold, run_time):
    duration = run_time * 60 * 60
    start_time = time.time()
    # Load data with memory mapping
    data = np.load(dataPath, mmap_mode='r')

    # Create a range of indices for combinations
    cb = range(len(data))
    mlp = []
    strength = []
    arr = np.zeros((k, data.shape[1]))
    st = True
    print('Start mining multipoles in ' + dataPath)
    # Iterate over all combinations of k indices
    for i in itertools.combinations(cb, int(k)):
        for j in range(k):
            arr[j] = data[i[j]]

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

def Brute_force_v2(data, nametag, k, threshold):
    mlp = []
    st = []
    eg = []
    cv = []
    lg = []

    arr = np.zeros((k, data.shape[1]))

    start_time = time.time()
    print(f'Start brute force at {start_time}')

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
        if out <= threshold:
            st.append(out)

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

    df = pd.DataFrame({
        'Multipoles': mlp,
        'Strength': st,
        'Linear Gain': lg,
        'Eigenvector': eg,
        'Covariance': cv,
    })
    df.to_csv(f'{nametag}.csv', index=False)
    print(f'Finished mining after {(time.time() - start_time)/60} minutes')

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

def linear_dependence (input):
    arr = statm.zscore(input, axis=1)
    x = np.corrcoef(arr.T, rowvar=False)
    x = np.nan_to_num(x)

    eigenvalues, eigenvectors = np.linalg.eig(x)
    min_variance_index = np.argmin(eigenvalues)
    min_variance_eigenvector = eigenvectors[:, min_variance_index]

    linear_dependence = 1 - np.var(np.dot(arr.T, min_variance_eigenvector))

    return linear_dependence

def k_pole_of_ts(datapath, tag, index, k, threshold):
    print(f'Begin finding {k}_pole of {index} in {tag}!')
    start_time = time.time()

    data = np.load(datapath, allow_pickle=True)
    list_index = range(len(data))
    list_index = np.delete(list_index, index)

    results = []
    for pred in combinations(list_index, k - 1):
        pred = np.array(pred)
        arr = np.concatenate(([data[index]], data[pred]), axis=0)

        ld = linear_dependence(arr)
        if ld > threshold:
            results.append({
                'Multipoles': [index] + pred.tolist(),
                'Strength': ld
            })

    output_file = f'result_{k}_pole_of_ts{index}_{tag}.csv'
    pd.DataFrame(results).to_csv(output_file, index=False)

    elapsed_time = (time.time() - start_time) / 60
    print(f'Finished finding {k}_pole after {elapsed_time:.2f} minutes')

