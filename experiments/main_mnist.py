import numpy as np
import logging
import pickle

from collections import defaultdict

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from pprint import pprint

from model_utils.experiment import Experiment

import vari.models.vae
import vari.datasets

from vari.models import get_default_model_config
from vari.utilities import get_device, log_sum_exp
from vari.inference import log_gaussian, DeterministicWarmup

import IPython

LOGGER = logging.getLogger()
ex = Experiment(name='OOD VAE')


@ex.config
def default_configuration():
    tag = 'ood-detection'

    dataset_name = 'MNISTBinarized'
    exclude_labels = []
    preprocess='dynamic'
    dataset_kwargs = dict(
        split='train',
        preprocess=preprocess,
        exclude_labels=exclude_labels,
    )

    n_epochs = 1000
    batch_size = 256
    importance_samples = 10
    learning_rate = 3e-4
    warmup_epochs = 0

    # VariationalAutoencoder, LadderVariationalAutoencoder, AuxilliaryVariationalAutoencoder, LadderVariationalAutoencoder
    # TODO VariationalAutoencoder should support multiple stochastic layers instead of separate HierarchicalVariationalAutoencoder class
    vae_type = 'VariationalAutoencoder'
    # vae_type = 'HierarchicalVariationalAutoencoder'
    # vae_type = 'AuxilliaryVariationalAutoencoder'
    # vae_type = 'LadderVariationalAutoencoder'
    # vae_kwargs = get_vae_kwargs(vae_type)

    device = get_device()
    seed = 0
    

@ex.automain
def run(device, dataset_name, dataset_kwargs, vae_type, n_epochs, batch_size, learning_rate, importance_samples,
        warmup_epochs, seed):

    # Print config, set threads and seed
    pprint(ex.current_run.config)
    if device == 'cpu':
        torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get dataset, model, optimizer and other
    train_dataset = getattr(vari.datasets, dataset_name)
    train_dataset = train_dataset(**dataset_kwargs)
    print(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=device == 'cuda')

    model_kwargs = get_default_model_config(vae_type, dataset_name)
    model = getattr(vari.models.vae, vae_type)
    model = model(
        x_dim=np.prod(train_dataset[0][0].shape),
        **model_kwargs
    )
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    deterministic_warmup = DeterministicWarmup(n=warmup_epochs)

    epoch = 0
    i_update = 0
    best_elbo = -1e10
    z_dim = model.z_dim if isinstance(model.z_dim, int) else model.z_dim[-1]
    p_z_samples = torch.randn(1000, z_dim).to(device)
    try:
        while epoch < n_epochs:
            model.train()
            total_elbo, total_kl, total_likelihood, total_kls = 0, 0, 0, defaultdict(lambda: 0)
            beta = next(deterministic_warmup)
            for b, (x, _) in tqdm(enumerate(train_loader), leave=False, total=len(train_loader), smoothing=0):
                x = x.to(device)
                
                optimizer.zero_grad()

                # Importance sampling
                x_iw = x.view(x.shape[0], np.prod(x.shape[1:]))
                x_iw = x_iw.repeat(1, importance_samples).view(-1, x_iw.shape[1])
                
                x, px_args = model(x_iw)
                likelihood = model.log_likelihood(x_iw, *px_args)
                kl_divergence = model.kl_divergence

                elbo = likelihood - beta * kl_divergence

                # Importance sampling
                elbo = log_sum_exp(elbo.view(-1, importance_samples, 1), axis=1, sum_op=torch.mean).view(-1, 1)  # (B, 1, 1)

                loss = - torch.mean(elbo)
                
                loss.backward()
                optimizer.step()
                
                likelihood = log_sum_exp(likelihood.view(-1, importance_samples, 1), axis=1, sum_op=torch.mean).view(-1, 1)  # (B, 1, 1)
                kl_divergence = log_sum_exp(kl_divergence.view(-1, importance_samples, 1), axis=1, sum_op=torch.mean).view(-1, 1)  # (B, 1, 1)
                kl_divergences = [log_sum_exp(kl_divergence.view(-1, importance_samples, 1), axis=1, sum_op=torch.mean).view(-1, 1)  for kl_divergence in model.kl_divergences]

                total_elbo += elbo.mean().item()
                total_likelihood += likelihood.mean().item()
                total_kl += kl_divergence.mean().item()
                for i, kl in enumerate(kl_divergences):
                    total_kls[i] += kl.mean().item()

                ex.log_scalar(f'(batch) ELBO log p(x)', elbo.mean().item(), step=i_update)
                ex.log_scalar(f'(batch) log p(x|z)', likelihood.mean().item(), step=i_update)
                ex.log_scalar(f'(batch) KL(q(z|x)||p(z))', kl_divergence.mean().item(), step=i_update)
                ex.log_scalar(f'(batch) ß * KL(q(z|x)||p(z))', beta * kl_divergence.mean().item(), step=i_update)
                for i, kl in enumerate(kl_divergences):
                    ex.log_scalar(f'(batch) KL on z{i}', kl.mean().item(), step=i_update)
                i_update += 1
                
            total_elbo /= len(train_loader)
            total_likelihood /= len(train_loader)
            total_kl /= len(train_loader)
                
            ex.log_scalar(f'ELBO log p(x)', total_elbo, step=epoch)
            ex.log_scalar(f'log p(x|z)', total_likelihood, step=epoch)
            ex.log_scalar(f'KL(q(z|x)||p(z))', total_kl, step=epoch)
            ex.log_scalar(f'ß * KL(q(z|x)||p(z))', beta * total_kl, step=epoch)
            for i, kl in total_kls.items():
                ex.log_scalar(f'KL on z_{i}', kl / len(train_loader), step=epoch)

            print(f'Epoch {epoch:3d} | ELBO {total_elbo: 2.4f} | log p(x|z) {total_likelihood: 2.4f} | KL {total_kl:2.4f} | ß*KL {beta * total_kl:2.4f}')

            if epoch % 10 == 0 and total_elbo > best_elbo and deterministic_warmup.is_done:
                best_elbo = total_elbo
                best_kl = total_kl
                best_likelihood = total_likelihood
                torch.save(model.state_dict(), f'{ex.models_dir()}/model_state_dict.pkl')
                px, px_args = model.sample(p_z_samples)
                px_args = [v.cpu().detach().numpy() for v in px_args]
                np.save(f'{ex.models_dir()}/epoch_{epoch}_model_samples', px_args)
                print(f'Epoch {epoch:3d} | Saved model at ELBO {total_elbo: 2.4f}')
            
            epoch += 1
    except KeyboardInterrupt:
        print('Interrupted experiment')

    return f'ELBO={best_elbo:2f}, p(x|z)={best_likelihood:2f}, KL(q||p)={best_kl:2f}'
