
import torch

from torch.utils.data import DataLoader

from vari.datasets import Spirals
from vari.models.vae import VariationalAutoencoder
from vari.utilities import get_device
from vari.inference import log_gaussian

import IPython

batch_size = 32

device = get_device()

train_dataset = Spirals(n_samples=1000, noise=0.05, radius=1)
test1_dataset = Spirals(n_samples=1000, noise=0.05, radius=1.5)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device=='cuda')
test_loader = DataLoader(test1_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device=='cuda')

model = VariationalAutoencoder([2, 2, [64, 64]])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(50):
    model.train()
    total_loss, total_kl, total_log_px = 0, 0, 0
    for x, _ in train_loader:
        x = x.to(device)
        #IPython.embed()

        px, px_mu, px_sigma = model(x)

        likelihood = log_gaussian(x, px_mu, px_sigma)
        #likelihood = -binary_cross_entropy(reconstruction, x)
        elbo = likelihood + model.kl_divergence
        
        # loss = - model.kl_divergence.mean()  # Goes to zero
        # loss = - likelihood.mean()  # Goes towards zero but slowly
        loss = - torch.mean(elbo)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_kl += model.kl_divergence.mean().item()
        total_log_px += likelihood.mean().item()

    print(f'Loss {total_loss / len(train_loader):.4f} | log p(x) {total_log_px / len(train_loader):.4f} | KL {total_kl / len(train_loader):.4f}')

