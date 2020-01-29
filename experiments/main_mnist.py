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
from vari.utilities import get_device, summary
from vari.inference import DeterministicWarmup

import IPython

LOGGER = logging.getLogger()
ex = Experiment(name='OOD VAE')


@ex.config
def default_configuration():
    tag = 'ood-detection'

    dataset_name = 'MNISTBinarized'
    exclude_labels = []
    preprocess = 'dynamic'
    dataset_kwargs = dict(
        preprocess=preprocess,
        exclude_labels=exclude_labels,
    )

    n_epochs = 1000
    batch_size = 256
    importance_samples = 10
    learning_rate = 3e-4
    warmup_epochs = 0

    # VariationalAutoencoder, LadderVariationalAutoencoder, AuxilliaryVariationalAutoencoder, LadderVariationalAutoencoder
    vae_type = 'VariationalAutoencoder'
    # vae_type = 'HierarchicalVariationalAutoencoder'
    # vae_type = 'AuxilliaryVariationalAutoencoder'
    # vae_type = 'LadderVariationalAutoencoder'

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
    dataset = getattr(vari.datasets, dataset_name)
    train_dataset = dataset(split='train', **dataset_kwargs)
    test_dataset = dataset(split='test', **dataset_kwargs)
    print(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device=='cuda')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device=='cuda')

    model_kwargs = get_default_model_config(vae_type, dataset_name)
    torch.save(model_kwargs, f'{ex.models_dir()}/model_kwargs.pkl')
    model = getattr(vari.models.vae, vae_type)
    model = model(**model_kwargs)
    model.to(device)
    print(model)
    x = train_dataset[0][0]
    summary(model, (np.prod(x.shape),))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    deterministic_warmup = DeterministicWarmup(n=warmup_epochs)

    epoch = 0
    i_update = 0
    best_elbo = -1e10
    if isinstance(model, vari.models.vae.HierarchicalVariationalAutoencoder):
        pz_samples = model.encoder[-1].distribution.get_prior().sample(torch.Size([1000]))
    else:
        pz_samples = model.encoder.distribution.get_prior().sample(torch.Size([1000]))

    try:
        while epoch < n_epochs:
            model.train()
            total_elbo, total_kl, total_likelihood, total_kls = 0, 0, 0, defaultdict(lambda: 0)
            beta = next(deterministic_warmup)
            for b, (x, _) in tqdm(enumerate(train_loader), leave=False, total=len(train_loader), smoothing=0):
                x = x.to(device)

                optimizer.zero_grad()

                x = x.view(x.shape[0], np.prod(x.shape[1:]))
                elbo, likelihood, kl_divergence = model.elbo(x, importance_samples=importance_samples, beta=beta)

                loss = - torch.mean(elbo)
                loss.backward()
                optimizer.step()

                total_elbo += elbo.mean().item()
                total_likelihood += likelihood.mean().item()
                total_kl += kl_divergence.mean().item()
                for k, v in model.kl_divergences.items():
                    total_kls[k] += v.mean().item()

                ex.log_scalar(f'(batch) ELBO log p(x)', elbo.mean().item(), step=i_update)
                ex.log_scalar(f'(batch) log p(x|z)', likelihood.mean().item(), step=i_update)
                ex.log_scalar(f'(batch) KL(q(z|x)||p(z))', kl_divergence.mean().item(), step=i_update)
                ex.log_scalar(f'(batch) ß * KL(q(z|x)||p(z))', beta * kl_divergence.mean().item(), step=i_update)
                for k, v in model.kl_divergences.items():
                    ex.log_scalar(f'(batch) KL for {k}', v.mean().item(), step=i_update)
                i_update += 1
                
            total_elbo /= len(train_loader)
            total_likelihood /= len(train_loader)
            total_kl /= len(train_loader)
                
            ex.log_scalar(f'ELBO log p(x)', total_elbo, step=epoch)
            ex.log_scalar(f'log p(x|z)', total_likelihood, step=epoch)
            ex.log_scalar(f'KL(q(z|x)||p(z))', total_kl, step=epoch)
            ex.log_scalar(f'ß * KL(q(z|x)||p(z))', beta * total_kl, step=epoch)
            for k, v in total_kls.items():
                ex.log_scalar(f'KL for {k}', v / len(train_loader), step=epoch)

            s = f'Epoch {epoch:3d} | ELBO {total_elbo: 2.4f} | log p(x|z) {total_likelihood: 2.4f} | KL {total_kl:2.4f} | ß*KL {beta * total_kl:2.4f}'
            if len(total_kls) > 1:
                for k, v in total_kls.items():
                    s += f' | KL {k} {v / len(train_loader):2.4f}'
            print(s)

            if epoch % 10 == 0 and total_elbo > best_elbo and deterministic_warmup.is_done:
                best_elbo = total_elbo
                best_kl = total_kl
                best_likelihood = total_likelihood
                torch.save(model.state_dict(), f'{ex.models_dir()}/model_state_dict.pkl')
                px = model.generate(z=pz_samples)
                np.save(f'{ex.models_dir()}/epoch_{epoch}_model_samples', px.mean.cpu().detach().numpy())
                print(f'Epoch {epoch:3d} | Saved model at ELBO {total_elbo: 2.4f}')

                model.eval()
                for iws in [1, importance_samples, 100]:
                    total_elbo, total_kl, total_likelihood, total_kls = 0, 0, 0, defaultdict(lambda: 0)
                    for b, (x, _) in enumerate(test_loader):
                        x = x.to(device)

                        optimizer.zero_grad()

                        x = x.view(x.shape[0], np.prod(x.shape[1:]))
                        elbo, likelihood, kl_divergence = model.elbo(x, importance_samples=iws, beta=1)

                        total_elbo += elbo.mean().item()
                        total_likelihood += likelihood.mean().item()
                        total_kl += kl_divergence.mean().item()
                        for k, v in model.kl_divergences.items():
                            total_kls[k] += v.mean().item()

                    total_elbo /= len(test_loader)
                    total_likelihood /= len(test_loader)
                    total_kl /= len(test_loader)

                    ex.log_scalar(f'[TEST] IW={iws} ELBO log p(x)', total_elbo, step=epoch)
                    ex.log_scalar(f'[TEST] IW={iws} log p(x|z)', total_likelihood, step=epoch)
                    ex.log_scalar(f'[TEST] IW={iws} KL(q(z|x)||p(z))', total_kl, step=epoch)
                    ex.log_scalar(f'[TEST] IW={iws} ß * KL(q(z|x)||p(z))', beta * total_kl, step=epoch)
                    for k, v in total_kls.items():
                        ex.log_scalar(f'[TEST] IW={iws} KL for {k}', v / len(train_loader), step=epoch)

                    s = f'[TEST] IW {iws:<3d} | Epoch {epoch:3d} | ELBO {total_elbo: 2.4f} | log p(x|z) {total_likelihood: 2.4f} | KL {total_kl:2.4f}'
                    if len(total_kls) > 1:
                        for k, v in total_kls.items():
                            s += f' | KL {k} {v / len(test_loader):2.4f}'
                    print(s)

            epoch += 1

    except KeyboardInterrupt:
        print('Interrupted experiment')
        return f'ELBO={best_elbo:2f}, p(x|z)={best_likelihood:2f}, KL(q||p)={best_kl:2f}'

    return f'ELBO={best_elbo:2f}, p(x|z)={best_likelihood:2f}, KL(q||p)={best_kl:2f}'
