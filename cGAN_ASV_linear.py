import argparse
import os
import numpy as np
import math

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, time_series_data, conditions):
        self.time_series_data = time_series_data
        self.conditions = conditions

    def __len__(self):
        return len(self.time_series_data)

    def __getitem__(self, idx):
        return self.time_series_data[idx], self.conditions[idx]


# Hyperparameters
noise_dim = 100
condition_dim = 150
output_dim = 165
batch_size = 256
learning_rate = 0.0002
num_epochs = 400


class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim=150, output_dim=165, embedding_dim=150):
        super(Generator, self).__init__()
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


class MiniBatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, intermediate_dim=50):
        super(MiniBatchDiscrimination, self).__init__()

        # Randomized tensor for learning minibatch discrimination
        self.T = nn.Parameter(torch.randn(in_features, out_features, intermediate_dim))
        self.out_features = out_features

    def forward(self, x):
        # Calculate minibatch features
        batch_size = x.size(0)
        M = x.mm(self.T.view(x.size(1), -1))  # Shape: (batch_size, out_features * intermediate_dim)
        M = M.view(batch_size, self.out_features, -1)  # Shape: (batch_size, out_features, intermediate_dim)

        # Compute differences across the batch
        M_diff = M.unsqueeze(0) - M.unsqueeze(1)
        abs_diff = torch.sum(torch.abs(M_diff), dim=3)

        # Apply negative exponential for minibatch features
        minibatch_features = torch.exp(-abs_diff).sum(dim=1)

        # Concatenate with the original input features
        return torch.cat([x, minibatch_features], dim=1)


class Discriminator(nn.Module):
    def __init__(self, input_dim=165, condition_dim=150, embedding_dim=150, minibatch_features=10):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(condition_dim, embedding_dim)

        # Updated minibatch discrimination layer
        self.minibatch = MiniBatchDiscrimination(
            input_dim + embedding_dim, minibatch_features)

        self.model = nn.Sequential(
            nn.Linear(input_dim + embedding_dim + minibatch_features, 1536),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1536, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, time_series, condition):
        y = self.embedding(condition)
        x = torch.cat([time_series, y], dim=1)
        x = self.minibatch(x)

        return self.model(x)


# Instantiate the models
generator = Generator(noise_dim, condition_dim, output_dim)
discriminator = Discriminator(output_dim, condition_dim)
generator.cuda()
discriminator.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()
# criterion = nn.MSELoss()
criterion.cuda()

# data
data = np.load("train_data/train_cGAN_ASV.npy", allow_pickle=True)
time_series_data, conditions = np.split(data, (data.shape[1] - 1,), axis=1)

dataset = TimeSeriesDataset(torch.tensor(time_series_data, dtype=torch.float32).cuda(),
                            torch.tensor(conditions, dtype=torch.long).cuda())
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def compute_gradient_penalty(discriminator, real_samples, fake_samples, condition):
    # Random weight for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1).cuda()
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # Get discriminator's output for interpolated samples
    d_interpolates = discriminator(interpolates, condition)

    # Calculate gradients with respect to the interpolated samples
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Calculate the gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


save_epoch = 20
# train loop
for epoch in range(num_epochs):
    for i, (real_time_series, condition) in enumerate(data_loader):
        batch_size = real_time_series.size(0)
        condition = condition.reshape(-1)

        # Real and fake labels
        real_labels = torch.ones(batch_size, 1).cuda()
        fake_labels = torch.zeros(batch_size, 1).cuda()

        # Train Discriminator
        optimizer_D.zero_grad()

        # Real time series
        d_loss_real = criterion(discriminator(real_time_series, condition), real_labels)

        # Fake time series
        noise = torch.randn(batch_size, noise_dim).cuda()
        fake_time_series = generator(noise, condition)
        d_loss_fake = criterion(discriminator(fake_time_series.detach(), condition), fake_labels)

        # gradient_penalty = compute_gradient_penalty(discriminator, real_time_series.data, fake_time_series.data, condition)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()

        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        outputs = discriminator(fake_time_series, condition)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()

        optimizer_G.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}] - d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}')

    if (epoch) % save_epoch == 0:
        torch.save(generator.state_dict(), "models/G_ts_ASV_linear-%d.model" % (epoch))
        # torch.save(discriminator.state_dict(), "models/D_ts_ASV_linear-%d.model" % (epoch))