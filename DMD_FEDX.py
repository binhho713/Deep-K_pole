"""
Main code for training and evaluating FedX.

"""

import argparse
import copy
import datetime
import json
import os
import random
import re
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from losses import js_loss, nt_xent
from fed_model import init_nets, init_mappingNet
from utils import get_dataloader, mkdirs, partition_data,  test_linear_fedX, set_logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", help="neural network used in training")
    parser.add_argument("--dataset", type=str, default="TS_Demixing", help="dataset used for train ing")
    parser.add_argument("--net_config", type=lambda x: list(map(int, x.split(", "))))
    parser.add_argument("--partition", type=str, default="noniid", help="the data partitioning strategy")
    parser.add_argument("--batch_size", type=int, default=64, help="total sum of input batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=10, help="number of local epochs")
    parser.add_argument("--n_parties", type=int, default=10, help="number of workers in a distributed cluster")
    parser.add_argument("--comm_round", type=int, default=50, help="number of maximum communication round")
    parser.add_argument("--init_seed", type=int, default=0, help="Random seed")
    parser.add_argument("--datadir", type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument("--reg", type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument("--logdir", type=str, required=False, default="./logs/", help="Log directory path")
    parser.add_argument("--modeldir", type=str, required=False, default="./client_models/models/", help="Model directory path")
    parser.add_argument(
        "--beta", type=float, default=0.5, help="The parameter for the dirichlet distribution for data partitioning"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to run the program")
    parser.add_argument("--optimizer", type=str, default="sgd", help="the optimizer")
    parser.add_argument("--out_dim", type=int, default=512, help="the output dimension for the projection layer")
    parser.add_argument("--temperature", type=float, default=0.1, help="the temperature parameter for contrastive loss")
    parser.add_argument("--tt", type=float, default=0.1, help="the temperature parameter for js loss in teacher model")
    parser.add_argument("--ts", type=float, default=0.1, help="the temperature parameter for js loss in student model")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="how many clients are sampled in each round")
    args = parser.parse_args()
    return args

nums = 10
gan1_size = 171
gan2_size = 1500
latent_dim = 100
ts_size1 = 108
ts_size2 = 325
n_classes = 10
m_lr = 0.0001
batch_size = 128

torch.autograd.set_detect_anomaly(True)

class Generator1(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=171, output_dim=108, embedding_dim=171):
        super(Generator1, self).__init__()
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

class Generator2(nn.Module):
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

gen_labels = np.load('./client_data/label.npy')

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def transform(x):
    tx = np.zeros((x.shape[0], x.shape[1]))
    noise = np.random.normal(0, 0.01, (x.shape[0], x.shape[1]))
    mask = np.random.rand(x.shape[0], x.shape[1]) > 0.1
    tx = (x + noise) * mask

    return tx

def entropy(x):
    return torch.mean(-torch.sum(x * torch.log(x + 1e-9), dim=1), dim=0)

def train_net_fedx(
    net_id,
    net,
    global_net,
    train_dataloader,
    epochs,
    lr,
    args_optimizer,
    temperature,
    args,
    round,
    device="cpu",
):
    predictor = net['predictor']
    seperator = net['seperator']
    predictor.cuda()
    seperator.cuda()
    global_p = global_net['predictor']
    global_s = global_net['seperator']
    global_p.cuda()
    global_s.cuda()
    logger.info("Training network %s" % str(net_id))
    logger.info("n_training: %d" % len(train_dataloader))

    # Set optimizer
    if args_optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(predictor.parameters()) + list(seperator.parameters())), lr=lr, weight_decay=args.reg)
    elif args_optimizer == "amsgrad":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, list(predictor.parameters()) + list(seperator.parameters())), lr=lr, weight_decay=args.reg, amsgrad=True
        )
    elif args_optimizer == "sgd":
        # optimizer = optim.SGD(
        #     filter(lambda p: p.requires_grad, list(predictor.parameters()) + list(seperator.parameters())), lr=lr, momentum=0.9, weight_decay=args.reg
        # )
        optimizer0 = optim.SGD(
            filter(lambda p: p.requires_grad, predictor.parameters()), lr=lr, momentum=0.9, weight_decay=args.reg
        )
        optimizer1 = optim.SGD(
            filter(lambda p: p.requires_grad, seperator.parameters()), lr=lr, momentum=0.9, weight_decay=args.reg
        )

    predictor.train()
    seperator.train()
    global_p.eval()
    global_s.eval()

    cnt = 0
    check_freq = 2

    for epoch in range(epochs):
        epoch_loss_collector = []

        indice_arr = np.arange(train_dataloader.shape[0])
        np.random.shuffle(indice_arr)

        rand_indices = indice_arr[:batch_size]

        while len(indice_arr) >= batch_size:

            indices = indice_arr[:batch_size]
            indice_arr = indice_arr[batch_size:]

            x1 = train_dataloader[indices]
            x2 = transform(x1)
            if len(indice_arr) >= batch_size:
                random_x = train_dataloader[indice_arr[:batch_size]]
            else:
                random_x = train_dataloader[rand_indices]

            x1 = torch.tensor(x1)
            x2 = torch.tensor(x2)
            random_x = torch.tensor(random_x)

            all_x = torch.cat((x1, x2, random_x), dim=0).cuda()
            all_x = all_x.reshape(-1, 1, all_x.shape[1]).float()

            _, p_proj1, p_pred1 = predictor(all_x)
            _, s_proj1, s_pred1 = seperator(all_x)
            with torch.no_grad():
                _, p_proj2, p_pred2 = global_p(all_x)
                _, s_proj2, s_pred2 = global_s(all_x)

            p_pred1_original, p_pred1_pos, p_pred1_random = p_pred1.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)
            p_proj1_original, p_proj1_pos, p_proj1_random = p_proj1.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)
            p_proj2_original, p_proj2_pos, p_proj2_random = p_proj2.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)

            s_pred1_original, s_pred1_pos, s_pred1_random = s_pred1.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)
            s_proj1_original, s_proj1_pos, s_proj1_random = s_proj1.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)
            s_proj2_original, s_proj2_pos, s_proj2_random = s_proj2.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)

            # Contrastive losses (local, global)
            nt_local_pred = nt_xent(p_proj1_original, p_proj1_pos, args.temperature)
            nt_global_pred = nt_xent(p_pred1_original, p_proj2_pos, args.temperature)
            loss_nt_pred = nt_local_pred + nt_global_pred

            nt_local_sep = nt_xent(s_proj1_original, s_proj1_pos, args.temperature)
            nt_global_sep = nt_xent(s_pred1_original, s_proj2_pos, args.temperature)
            loss_nt_sep = nt_local_sep + nt_global_sep

            # Relational losses (local, global)
            js_global_pred = js_loss(p_pred1_original, p_pred1_pos, p_proj2_random, args.temperature, args.tt)
            js_local_pred = js_loss(p_proj1_original, p_proj1_pos, p_proj1_random, args.temperature, args.ts)
            loss_js_pred = js_global_pred + js_local_pred

            js_global_sep = js_loss(s_pred1_original, s_pred1_pos, s_proj2_random, args.temperature, args.tt)
            js_local_sep = js_loss(s_proj1_original, s_proj1_pos, s_proj1_random, args.temperature, args.ts)
            loss_js_sep = js_global_sep + js_local_sep

            loss_pred = loss_nt_pred + loss_js_pred
            loss_sep = loss_nt_sep + loss_js_sep

            optimizer0.zero_grad()
            loss_pred.backward()
            optimizer0.step()

            optimizer1.zero_grad()
            loss_sep.backward()
            optimizer1.step()

            epoch_loss_collector.append(loss_pred.item() + loss_sep.item())

        cnt += 1
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info("Epoch: %d Loss: %f" % (epoch, epoch_loss))

        if (cnt % check_freq == 0):

            print("Epoch: %d Loss: %f" % (epoch, epoch_loss))
            cnt = 0

    predictor.eval()
    seperator.eval()
    logger.info(" ** Round %d Training Complete **" % (round))


def local_train_net(
    nets,
    args,
    train_dl_local_dict,
    global_model=None,
    prev_model_pool=None,
    round=None,
    device="cpu",
):

    if global_model:
        global_model['predictor'].cuda()
        global_model['seperator'].cuda()

    start_time = time.time()
    logger.info("Start training at " + str(start_time))

    n_epoch = args.epochs
    for net_id, net in nets.items():
        logger.info("Training network %s. n_training: %d" % (str(net_id), 10000))
        print("Training network %s. n_training: %d" % (str(net_id), 10000))
        train_dl_local = train_dl_local_dict[net_id]
        train_net_fedx(
            net_id,
            net,
            global_model,
            train_dl_local,
            n_epoch,
            args.lr,
            args.optimizer,
            args.temperature,
            args,
            round,
            device=device,
        )

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(" ** Phase One Training Complete at %d after %d**" % (end_time, total_time))

    if global_model:
        global_model['predictor'].to("cpu")
        global_model['seperator'].to("cpu")

    return nets

def local_train(mNets, net, train_data, m_lr):
    n_epoch = 100

    embsep_net = net['seperator']
    embpred_net = net['predictor']
    embsep_net.eval()
    embpred_net.eval()

    start_time = time.time()
    logger.info("Start training at " + str(start_time))

    for net_id, mNet in mNets.items():

        logger.info("Training Local Model Of Client" + str(net_id))

        if net_id < 5:
            ts_size = ts_size1
            decoder = Generator1()
            if torch.cuda.is_available():
                decoder = decoder.cuda()
        else:
            ts_size = ts_size2
            decoder = Generator2()
            if torch.cuda.is_available():
                decoder = decoder.cuda()

        decoder.eval()
        decoder.load_state_dict(torch.load(decoder_files[net_id]))
        decoder.eval()

        sep_net = mNet['seperator']
        pred_net = mNet['predictor']
        sep_net.train()
        pred_net.train()
        sep_net.cuda()
        pred_net.cuda()

        client_data = train_data[net_id]
        labels = np.tile(gen_labels[net_id], batch_size)[:640]
        labels = torch.tensor(labels).long().cuda()

        check_freq = 100

        for epoch in range(n_epoch):
            epoch_loss_collector = []
            running_loss = 0
            cnt = 0

            indice_arr = np.arange(client_data.shape[0])
            np.random.shuffle(indice_arr)

            while len(indice_arr) >= batch_size:

                indices = indice_arr[:batch_size]
                indice_arr = indice_arr[batch_size:]

                x = client_data[indices]
                x = torch.tensor(x.reshape(-1, 1, x.shape[1])).float().cuda()

                optimizer = torch.optim.Adam(list(pred_net.parameters()) + list(sep_net.parameters()), lr=m_lr)

                e_sep, _, _ = embsep_net(x)
                e_pred, _, _ = embpred_net(x)

                e_sep = torch.tensor(e_sep).float().cuda()
                e_pred = torch.tensor(e_pred).float().cuda()

                sep = sep_net(e_sep)
                pred = pred_net(e_pred)

                sep = torch.tensor(sep).float().cuda()

                rec = decoder(sep.view(-1, 100), labels)  # bs*10, 1, 1024
                gen_ts = torch.reshape(gen_ts, (-1, 1, ts_size))
                rec = rec.permute(1, 2, 0) * pred.view(-1)  # 1,1024,bs*10
                rec = rec.permute(2, 0, 1).view(batch_size, n_classes, 1, ts_size)  # bs, 10, 1, 1024
                rec = torch.sum(rec, dim=1)  # bs, 1, 1024
                cri = torch.nn.MSELoss()
                loss_rec = cri(x, rec)  # can doi all_x

                # k_sparity constrain
                c = np.log(3) - 1e-6
                c = torch.tensor(c).float()
                k_sparity = torch.maximum(entropy(pred), c).float()

                loss = loss_rec + 0.1 * k_sparity

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                running_loss += loss_rec.item()
                cnt += 1
                epoch_loss_collector.append(loss_rec.item())

                if (cnt % check_freq == 0):
                    print("Epoch: %d Loss: %f" % (epoch, running_loss/cnt))

                    torch.save(sep_net.state_dict(), args.modeldir + "local_sep_mapping_model_check_%d" %(net_id))
                    torch.save(pred_net.state_dict(), args.modeldir + "local_pred_mapping_model_check_%d" %(net_id))

                    cnt = 0
                    running_loss = 0

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info("Epoch: %d Loss: %f" % (epoch, epoch_loss))



        sep_net.train()
        pred_net.train()

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(" ** Phase Two Training Complete at %d after %d**" % (end_time, total_time))

    return mNets

if __name__ == "__main__":
    args = get_args()

    # Create directory to save log and model
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    argument_path = f"{args.dataset}-{args.batch_size}-{args.n_parties}-{args.temperature}-{args.tt}-{args.ts}-{args.epochs}_arguments-%s.json" % datetime.datetime.now().strftime(
        "%Y-%m-%d-%H%M-%S"
    )

    # Save arguments
    with open(os.path.join(args.logdir, argument_path), "w") as f:
        json.dump(str(args), f)

    device = torch.device(args.device)

    # Set logger
    logger = set_logger(args)
    logger.info(device)

    # Set seed
    set_seed(args.init_seed)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []

    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    # Initializing net from each local party.
    logger.info("Initializing nets")

    nets = init_nets(args.net_config, args.n_parties, args, device="cpu")
    global_models = init_nets(args.net_config, 1, args, device="cpu")

    global_model = global_models[0]
    n_comm_rounds = args.comm_round

    ####### Dataset setting - under working

    folder_path = './client_data'
    pattern = re.compile(r'train_client_(\d+).npy')

    train_data_local = {}

    for file in os.listdir(folder_path):
        match = pattern.match(file)
        if match:
            index = int(match.group(1))
            file_path = os.path.join(folder_path, file)
            memmap = np.load(file_path, mmap_mode='r')
            train_data_local[index] = memmap

    for idx in range(len(train_data_local)):
        print(idx, train_data_local[idx].filename)

    model_path = './client_models/decoder'
    pattern = re.compile(r'G_ts_client_(\d+)-180.model')
    decoder_files = {}

    for file in os.listdir(model_path):
        match = pattern.match(file)
        if match:
            index = int(match.group(1))
            decoder_files[index] = os.path.join(model_path, file)

    for idx in range(len(decoder_files)):
        print(idx, decoder_files[idx])

    #######

    # Main training communication loop.
    for round in range(n_comm_rounds):
        logger.info("in comm round:" + str(round))
        party_list_this_round = party_list_rounds[round]

        # Download global model from (virtual) central server
        global_pred_w = global_model['predictor'].state_dict()
        global_sep_w = global_model['seperator'].state_dict()

        nets_this_round = {k: nets[k] for k in party_list_this_round}
        for net in nets_this_round.values():
            net['predictor'].load_state_dict(global_pred_w)
            net['seperator'].load_state_dict(global_sep_w)

        # Train local model with local data

        local_train_net(
            nets_this_round,
            args,
            train_data_local,
            round=round,
            global_model=global_model,
            device=device,
        )

        # total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
        # fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]

        total_data_points = 10 * 10000
        fed_avg_freqs = [10000 / total_data_points for r in range(args.n_parties)]

        # Averaging the local models' parameters to get global model
        for net_id, net in enumerate(nets_this_round.values()):
            net_pred_para = net['predictor'].state_dict()
            net_sep_para = net['seperator'].state_dict()
            if net_id == 0:
                for key in net_pred_para:
                    global_pred_w[key] = net_pred_para[key] * fed_avg_freqs[net_id]
                for key in net_sep_para:
                    global_sep_w[key] = net_sep_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_pred_para:
                    global_pred_w[key] += net_pred_para[key] * fed_avg_freqs[net_id]
                for key in net_sep_para:
                    global_sep_w[key] += net_sep_para[key] * fed_avg_freqs[net_id]

        global_model['predictor'].load_state_dict(copy.deepcopy(global_pred_w))
        global_model['seperator'].load_state_dict(copy.deepcopy(global_sep_w))
        global_model['predictor'].cuda()
        global_model['seperator'].cuda()

        torch.save(global_model['seperator'].state_dict(), args.modeldir + "global_sep_model_" + str(round))
        torch.save(global_model['predictor'].state_dict(), args.modeldir + "global_pred_model_" + str(round))

    #torch.save(nets[0]['seperator'].state_dict(), args.modeldir + "local_sep_model0" + args.log_file_name + ".pth")
    #torch.save(nets[0]['predictor'].state_dict(), args.modeldir + "local_pred_model0" + args.log_file_name + ".pth")

    logger.info("Start Second Phase Training")

    global_model['predictor'].load_state_dict(torch.load(f"./client_models/models/global_pred_model_{n_comm_rounds-1}"))
    global_model['seperator'].load_state_dict(torch.load(f"./client_models/models/global_sep_model_{n_comm_rounds-1}"))
    global_model['predictor'].cuda()
    global_model['seperator'].cuda()

    mNets = init_mappingNet(args.n_parties, args, device="cpu")
    local_train(mNets, global_model, train_data_local, m_lr)

