
import torch
import pickle

from torch.utils.data import DataLoader

from model_utils.experiment import Experiment

from vari.datasets import Spirals
from vari.models.vae import VariationalAutoencoder, AuxilliaryVariationalAutoencoder
from vari.utilities import get_device, log_sum_exp
from vari.inference import log_gaussian

import IPython

ex = Experiment(name='OOD AVAE')


@ex.config
def default_configuration():
    tag = 'ood-detection'
    n_epochs = 2000
    batch_size = 32
    importance_samples = 10
    learning_rate = 3e-4
    device = get_device()


@ex.automain
def run(device, n_epochs, batch_size, learning_rate, importance_samples):
    train_dataset = Spirals(n_samples=1000, noise=0.05, radius=1)
    test1_dataset = Spirals(n_samples=1000, noise=0.05, radius=1.5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=device=='cuda')
    test_loader = DataLoader(test1_dataset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=device=='cuda')

    model = AuxilliaryVariationalAutoencoder(x_dim=2, z_dim=2, a_dim=3, h_dims=[64, 64, 64])
    # model = VariationalAutoencoder(x_dim=2, z_dim=2, h_dims=[64, 64, 64])
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch = 0
    best_elbo = 1e10
    # for epoch in range(n_epochs):
    while epoch < n_epochs:
        model.train()
        total_elbo, total_kl, total_log_px = 0, 0, 0
        for b, (x, _) in enumerate(train_loader):
            x = x.to(device)

            x = x.repeat(1, importance_samples).view(-1, x.shape[1])

            px, px_mu, px_sigma = model(x)
            kl_divergence = model.kl_divergence

            likelihood = log_gaussian(x, px_mu, px_sigma)
            elbo = likelihood - kl_divergence
            
            # Importance sampling
            # elbo = elbo.view(-1, importance_samples, 1)  # (B, IW, 1)
            # elbo = log_sum_exp(elbo, axis=1, sum_op=torch.sum)  # (B, 1, 1)
            # elbo = elbo.view(-1, 1)  # (B, 1)
            elbo = log_sum_exp(elbo.view(-1, importance_samples, 1), axis=1, sum_op=torch.sum).view(-1, 1)  # (B, 1, 1)
            kl_divergence = log_sum_exp(kl_divergence.view(-1, importance_samples, 1), axis=1, sum_op=torch.sum).view(-1, 1)  # (B, 1, 1)
            likelihood = log_sum_exp(likelihood.view(-1, importance_samples, 1), axis=1, sum_op=torch.sum).view(-1, 1)  # (B, 1, 1)
            
            loss = - torch.mean(elbo)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_elbo += loss.item()
            total_kl += model.kl_divergence.mean().item()
            total_log_px += likelihood.mean().item()
            
            # ex.log_scalar(f'total_loss_b', loss.item(), step=epoch)
            
        total_elbo = total_elbo / len(train_loader)
        total_log_px = total_log_px / len(train_loader)
        total_kl = total_kl / len(train_loader)
            
        ex.log_scalar(f'ELBO', total_elbo, step=epoch)
        ex.log_scalar(f'log p(x|z)', total_log_px, step=epoch)
        ex.log_scalar(f'kl(q|p)', total_kl, step=epoch)

        print(f'Epoch {epoch} | ELBO {total_elbo:.4f} | log p(x|z) {total_log_px:.4f} | KL {total_kl:.4f}')

        if epoch % 10 == 0 and total_elbo < best_elbo:
            best_elbo = total_elbo
            torch.save(model.state_dict(), f'{ex.models_dir()}/model_state_dict.pkl')
        
        epoch += 1
