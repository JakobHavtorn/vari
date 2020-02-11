import numpy as np
import logging
import pickle

from collections import defaultdict

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from pprint import pprint
from sacred import SETTINGS

from model_utils.experiment import Experiment

import vari.models.vae
import vari.datasets

from vari.layers import GaussianLayer
from vari.models import build_dense_vae
from vari.utilities import get_device, summary
from vari.inference import DeterministicWarmup

import IPython


SETTINGS.CONFIG.READ_ONLY_CONFIG = False
LOGGER = logging.getLogger()
ex = Experiment(name='OOD VAE')


@ex.config
def default_configuration():
    tag = 'ood-detection'

    dataset_name = 'MNISTBinarized'
    dataset_kwargs = dict(
        preprocess='dynamic',
        exclude_labels=[],
    )

    n_epochs = 1000
    batch_size = 256
    importance_samples = 10
    learning_rate = 3e-4
    warmup_epochs = 0
    free_nats = -np.inf
    
    model_kwargs = dict(
        x_dim=784,
        z_dim=[5, 2],
        h_dim=[[512, 512], [256, 256]],
        activation=torch.nn.LeakyReLU
    )

    test_importance_samples = set([1, importance_samples, 100])
    device = get_device()
    seed = 0


@ex.config
def dependent_configuration(model_kwargs):
    model_kwargs['encoder_distribution'] = ['GaussianLayer'] * len(model_kwargs['z_dim'])
    model_kwargs['decoder_distribution'] = ['GaussianLayer'] * (len(model_kwargs['z_dim']) - 1) + ['BernoulliLayer']


@ex.automain
def run(device, dataset_name, dataset_kwargs, model_kwargs, n_epochs, batch_size, learning_rate, importance_samples,
        warmup_epochs, free_nats, test_importance_samples, seed):

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

    model, model_kwargs = build_dense_vae(**model_kwargs)
    torch.save(model_kwargs, f'{ex.models_dir()}/model_kwargs.pkl')
    model.to(device)
    print(model)
    summary(model, (np.prod(train_dataset[0][0].shape),))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    deterministic_warmup = DeterministicWarmup(n=warmup_epochs)

    epoch = 0
    i_update = 0
    best_elbo = -1e10
    pz_samples = model.encoder[-1].distribution.prior.sample(torch.Size([1000]))

    try:
        while epoch < n_epochs:
            model.train()
            total_elbo, total_kl, total_likelihood, total_kls = 0, 0, 0, defaultdict(lambda: 0)
            total_elbo_1iw, total_kl_1iw, total_likelihood_1iw, total_kls_1iw = 0, 0, 0, defaultdict(lambda: 0)
            beta = next(deterministic_warmup)
            for b, (x, _) in enumerate(train_loader):
                x = x.to(device)

                optimizer.zero_grad()

                x = x.view(x.shape[0], np.prod(x.shape[1:]))
                # elbo, likelihood, kl_divergence = model.elbo(x, importance_samples=importance_samples, beta=beta)
                elbo, likelihood, kl_divergence = model.elbo(x, importance_samples=importance_samples, beta=beta,
                                                             free_nats=free_nats, reduce_importance_samples=False)
                
                kl_divergences_1iw = {k: v[0, ...] for k, v in model.kl_divergences.items()}
                elbo_1iw, likelihood_1iw, kl_divergence_1iw = elbo[0, ...], likelihood[0, ...], kl_divergence[0, ...]
                kl_divergences = model.kl_divergences
                elbo, likelihood, kl_divergence = model.reduce_importance_samples(elbo, likelihood, kl_divergences)

                loss = - torch.mean(elbo)
                loss.backward()
                optimizer.step()

                total_elbo += elbo.mean().item()
                total_likelihood += likelihood.mean().item()
                total_kl += kl_divergence.mean().item()
                for k, v in kl_divergences.items():
                    total_kls[k] += v.mean().item()
                total_elbo_1iw += elbo_1iw.mean().item()
                total_likelihood_1iw += likelihood_1iw.mean().item()
                total_kl_1iw += kl_divergence_1iw.mean().item()
                for k, v in kl_divergences_1iw.items():
                    total_kls_1iw[k] += v.mean().item()

                # ex.log_scalar(f'(batch) ELBO log p(x)', elbo.mean().item(), step=i_update)
                # ex.log_scalar(f'(batch) log p(x|z)', likelihood.mean().item(), step=i_update)
                # ex.log_scalar(f'(batch) KL(q(z|x)||p(z))', kl_divergence.mean().item(), step=i_update)
                # ex.log_scalar(f'(batch) ß * KL(q(z|x)||p(z))', beta * kl_divergence.mean().item(), step=i_update)
                # for k, v in kl_divergences.items():
                #     ex.log_scalar(f'(batch) KL for {k}', v.mean().item(), step=i_update)
                i_update += 1
                
            total_elbo /= len(train_loader)
            total_likelihood /= len(train_loader)
            total_kl /= len(train_loader)
            total_elbo_1iw /= len(train_loader)
            total_likelihood_1iw /= len(train_loader)
            total_kl_1iw /= len(train_loader)

            ex.log_scalar(f'IW {importance_samples} log p(x)', total_elbo, step=epoch)
            ex.log_scalar(f'IW {importance_samples} log p(x|z)', total_likelihood, step=epoch)
            ex.log_scalar(f'IW {importance_samples} KL(q||p)', total_kl, step=epoch)
            ex.log_scalar(f'IW {importance_samples} ß·KL(q||p)', beta * total_kl, step=epoch)
            for k, v in total_kls.items():
                ex.log_scalar(f'IW {importance_samples} KL for {k}', v / len(train_loader), step=epoch)
            
            if importance_samples != 1:
                ex.log_scalar(f'IW 1 log p(x)', total_elbo_1iw, step=epoch)
                ex.log_scalar(f'IW 1 log p(x|z)', total_likelihood_1iw, step=epoch)
                ex.log_scalar(f'IW 1 KL(q||p)', total_kl_1iw, step=epoch)
                ex.log_scalar(f'IW 1 ß·KL(q||p)', beta * total_kl_1iw, step=epoch)
                for k, v in total_kls_1iw.items():
                    ex.log_scalar(f'IW 1 KL for {k}', v / len(train_loader), step=epoch)
                
            s = f'Epoch {epoch:3d} | ELBO {total_elbo: 2.4f} | log p(x|z) {total_likelihood: 2.4f} | KL {total_kl:2.4f} | ß*KL {beta * total_kl:2.4f}'
            if len(total_kls) > 1:
                for k, v in total_kls.items():
                    s += f' | KL {k} {v / len(train_loader):2.4f}'
            print(s)

            if epoch % 50 == 0:
                # Evaluate model
                model.eval()
                for iws in sorted(test_importance_samples):
                    total_elbo, total_kl, total_likelihood, total_kls = 0, 0, 0, defaultdict(lambda: 0)
                    # Dynamic batch size depending on the number of importance samples
                    test_loader = DataLoader(test_dataset, batch_size=(batch_size * 3) // iws + 1, shuffle=True, num_workers=2, pin_memory=device=='cuda')
                    for b, (x, _) in enumerate(test_loader):
                        x = x.to(device)

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

                    ex.log_scalar(f'[TEST] IW={iws} log p(x)', total_elbo, step=epoch)
                    ex.log_scalar(f'[TEST] IW={iws} log p(x|z)', total_likelihood, step=epoch)
                    ex.log_scalar(f'[TEST] IW={iws} KL(q||p)', total_kl, step=epoch)
                    ex.log_scalar(f'[TEST] IW={iws} ß·KL(q||p)', beta * total_kl, step=epoch)
                    for k, v in total_kls.items():
                        ex.log_scalar(f'[TEST] IW={iws} KL for {k}', v / len(test_loader), step=epoch)

                    s = f'[TEST] IW {iws:<3d} | Epoch {epoch:3d} | ELBO {total_elbo: 2.4f} | log p(x|z) {total_likelihood: 2.4f} | KL {total_kl:2.4f}'
                    if len(total_kls) > 1:
                        for k, v in total_kls.items():
                            s += f' | KL {k} {v / len(test_loader):2.4f}'
                    print(s)
                    
                # Dump encodings and decodings
                with torch.no_grad():
                    px = model.generate(z=pz_samples)
                    np.save(f'{ex.models_dir()}/epoch_{epoch}_model_samples', px.mean.cpu().detach().numpy())

                    x, y = next(iter(test_loader))
                    x = x.view(x.shape[0], np.prod(x.shape[1:]))
                    latents = model.encode(x.to(device))
                    px = model.decode(latents)
                    torch.save(latents, f'{ex.models_dir()}/epoch_{epoch}_model_latents.pkl')
                    torch.save(px, f'{ex.models_dir()}/epoch_{epoch}_model_outputs.pkl')
                    torch.save({'x': x, 'y': y}, f'{ex.models_dir()}/epoch_{epoch}_model_inputs.pkl')

                # Save model
                if total_elbo > best_elbo and deterministic_warmup.is_done:
                    best_elbo = total_elbo
                    best_kl = total_kl
                    best_likelihood = total_likelihood
                    torch.save(model.state_dict(), f'{ex.models_dir()}/model_state_dict.pkl')
                    print(f'Epoch {epoch:3d} | Saved model at ELBO {total_elbo: 2.4f}')
                    
            epoch += 1

    except KeyboardInterrupt:
        print('Interrupted experiment')
        return f'ELBO={best_elbo:.2f}, p(x|z)={best_likelihood:.2f}, KL(q||p)={best_kl:.2f}'

    return f'ELBO={best_elbo:.2f}, p(x|z)={best_likelihood:.2f}, KL(q||p)={best_kl:.2f}'
