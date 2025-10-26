import argparse
import os, sys
import numpy as np
import math
import scipy.io

from sklearn.metrics import accuracy_score

from torch.autograd import Variable
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import logging
import yaml
import argparse

from encoder_ASV import *
from helpers import *

parser = argparse.ArgumentParser(description="Load model config")
parser.add_argument("--k_query", type=int, default=3, help="Number of query samples")
parser.add_argument("--model_ver", type=int, default=200, help="Model version")
parser.add_argument("--epoch", type=int, default=200, help="Training epoches")
parser.add_argument("--is_training", type=int, default=1, help="training flag")

args = parser.parse_args()

with open("config.yaml", "r") as f:
    raw = yaml.safe_load(f)

config = raw["ASV"]

batch_size = config["batch_size"]
latent_dim = config["latent_dim"]
n_classes = config["n_classes"]
ts_size = config["ts_size"]
nums = config["nums"]
k_query = args.k_query
model_ver = args.model_ver
epoch = args.epoch
is_training = args.is_training

use_cuda = torch.cuda.is_available()

class Generator(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=150, output_dim=325, embedding_dim=150):
        super(Generator, self).__init__()
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

# Init generator & load pretrained generative decoder
generator = Generator()
if torch.cuda.is_available():
    print("use cuda")
    generator = generator.cuda()
generator.load_state_dict(torch.load(f"models/G_ts_ASV_linear-{model_ver}.model"))
generator.eval()

# Init seperator and predictor
sep = resnet18(predictor=False)
pred = resnet18(predictor=True)

if use_cuda:
    sep = sep.cuda()
    pred = pred.cuda()
    print("use_cuda")


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


# Load data
trainData = np.load("train_cGAN_ASV.npy", allow_pickle=True)
time_series, _ = np.split(trainData, (trainData.shape[1]-1,), axis=1)
arrInx = np.arange(0, time_series.shape[0], 1)
np.random.shuffle(arrInx)
n_timeSeries = len(time_series)
gen_labels = torch.LongTensor(np.arange(nums * batch_size, dtype="int32") % nums)
if use_cuda:
    gen_labels = gen_labels.cuda()


# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)


pred_path = f"./models/pred_DMD_ASV_{k_query}_"
sep_path = f"./models/sep_DMD_ASV_{k_query}_"

check_freq = 100
save_freq = 20
lr = 0.00001

logger = setup_logger(f'DMD_ASV_train_{k_query}_logger', f'DMD_ASV_training_{k_query}.log')

# ----------
# Training
# ----------
start_time = time.time()
logger.info("Start training at " + str(start_time))

if is_training:
    # ----------
    # Training
    # ----------
    for _epoch_ in range(epoch):

        ts_mix_lst = []
        running_loss = 0
        k_loss = 0
        cnt = 0

        for idx in range(n_timeSeries):

            ts_mix = time_series[arrInx[idx]][:ts_size]  # 1, 305
            ts_mix_lst.append(ts_mix)

            if (len(ts_mix_lst) == batch_size):
                ts_mix = np.concatenate(ts_mix_lst, axis=0)  # bs * 305
                ts_mix = Variable(torch.tensor(ts_mix.reshape(-1, 1, ts_size)).float(), requires_grad=False)  # bs, 305
                if use_cuda:
                    ts_mix = ts_mix.cuda()

                labels_distribution = pred(ts_mix)  # bs, 150
                if (use_cuda):
                    z = sep(torch.tensor(ts_mix).float()).cuda()  # bs, 1, 150, 305
                else:
                    z = sep(torch.tensor(ts_mix).float())

                optimizer = torch.optim.Adam(list(pred.parameters()) + list(sep.parameters()), lr=lr)

                # generate image
                gen_ts = generator(z.view(-1, 100), gen_labels)  # bs*150, 1, 305
                gen_ts = torch.reshape(gen_ts, (-1, 1, ts_size))
                gen_ts = gen_ts.permute(1, 2, 0) * labels_distribution.view(-1)  # 1, 305, bs*150
                gen_ts = gen_ts.permute(2, 0, 1).view(batch_size, n_classes, 1, ts_size)  # bs, 150, 1, 305
                gen_ts = torch.sum(gen_ts, dim=1)  # bs, 1, 1024

                # reconstruct loss
                loss_rec = 0
                cri = torch.nn.MSELoss()
                loss_rec = cri(ts_mix, gen_ts)

                # k_sparity constrain
                c = np.log(k_query - 1) - 1e-4
                c = torch.tensor(c).float()
                k_sparity = torch.maximum(torch.zeros(1).cuda(), helpers.entropy(labels_distribution) - c).float()

                loss = loss_rec + 0.1 * k_sparity
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                ts_mix_lst = []
                running_loss += loss_rec.item()
                k_loss += k_sparity.item()
                cnt += 1

                # save model
                if (cnt % check_freq == 0):
                    print("#epoch = %d, data = %d, running_loss = %f, k_loss= %f" % (
                    _epoch_, cnt * batch_size, running_loss / cnt, k_loss/cnt))

                    logger.info("#epoch = %d, data = %d, running_loss = %f, k_loss= %f" % (
                        _epoch_, cnt * batch_size, running_loss / cnt, k_loss / cnt))

                    save_model(pred, pred_path)
                    save_model(sep, sep_path)

                    running_label_acc = 0
                    running_loss = 0
                    cnt = 0

        if(_epoch_ % save_freq == 0):
            save_model(pred, pred_path + str(_epoch_) + '.model')
            save_model(sep, sep_path + str(_epoch_) + '.model')

    end_time = time.time()
    total_time = end_time - start_time
    logger.info("Training completed at %d after %d" % (end_time, total_time))

else:
# ----------
# Testing
# ----------

    # --- load & normalize ---
    inTs = np.load('base_ts_150_ASV.npy', allow_pickle=True)
    inTs = statm.zscore(inTs, axis=1)  # per-series z-score
    samples, length = inTs.shape
    eval_label = np.arange(samples)

    # --- model ---
    pred0 = encoder_ASV.resnet18(predictor=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(f"models/pred_DMD_ASV_{k_query}_{model_ver}.model",
                       map_location=device)
    pred0.load_state_dict(state)
    pred0.eval().to(device)

    k_pole = int(k_query)
    top = 10

    mlps, sts = [], []

    with torch.no_grad():
        for i in range(samples):
            test_data = inTs[i]  # shape: (length,)

            # (1, 1, length) tensor
            test_int = torch.tensor(test_data.reshape(1, 1, length), dtype=torch.float32, device=device)

            # forward
            predict = pred0(test_int)  # shape: (1, N)
            pred_labels = predict.detach().cpu().numpy()[0]  # (N,)

            # how many partners to pick (excluding i)
            # after you computed `pred_labels` and set k_pole/top
            pick = max(0, k_pole - 1)
            if pick == 0:
                # trivial case: only the test series itself
                arr = np.zeros((1, length))
                arr[0] = test_data
                mlps.append([np.array([i], dtype=int)])
                sts.append(0.0)
                continue

            # Your helper returns: List[Tuple[int, ...]]
            combos = helpers.top_m_combinations(pred_labels[0], k=pick, exclude_index=i, m=top)

            mlp = []
            st = []
            for combo in combos:
                # combo is a tuple of indices in original array space (already mapped by your helper)
                arr = np.zeros((k_pole, length))
                arr[0] = test_data
                for j in range(1, k_pole):
                    arr[j] = inTs[int(combo[j - 1])]

                # z-score per row
                arr = statm.zscore(arr, axis=1)

                # corr on columns (k_pole vars, length samples)
                X = arr.T
                C = np.corrcoef(X, rowvar=False)
                C = np.nan_to_num(C)

                # symmetric eigen is safer with eigh
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                vmin = eigenvectors[:, np.argmin(eigenvalues)]  # (k_pole,)

                # project rows onto min-variance direction
                out_series = np.tensordot(vmin, arr, axes=(0, 0))  # (length,)
                out = float(np.var(out_series))

                # record multipole indices (first = i, rest = combo)
                ml = np.zeros((k_pole,), dtype=int)
                ml[0] = i
                ml[1:] = np.array(combo, dtype=int)
                mlp.append(ml)
                st.append(out)

            mlps.append(mlp)
            sts.append(float(np.mean(st)) if st else 0.0)

    # save
    df = pd.DataFrame({
        'Multipoles': mlps,
        'Strength': sts,
    })
    df.to_excel(f'results/test_DMD_ASV_{k_pole}_pole_top{top}.xlsx', index=False)