import numpy as np
import torch
import pickle

from torch.utils.data import DataLoader
from pprint import pprint

from model_utils.experiment import Experiment

import vari.models.vae

from vari.datasets import Spirals, Moons
from vari.utilities import get_device, log_sum_exp
from vari.inference import log_gaussian, DeterministicWarmup

import IPython

ex = Experiment(name='OOD VAE')


@ex.config
def default_configuration():
    tag = 'ood-detection'
    n_epochs = 300
    batch_size = 32
    importance_samples = 10
    learning_rate = 3e-4
    dataset = 'moons'  # 'moons'
    # vae_type = 'DeepVariationalAutoencoder'  # LadderVariationalAutoencoder, AuxilliaryVariationalAutoencoder, LadderVariationalAutoencoder
    # vae_kwargs = dict(
    #     x_dim=2,
    #     z_dim=[2, 2],
    #     h_dim=[64, 64]
    # )
    vae_type = 'VariationalAutoencoder'  # LadderVariationalAutoencoder, AuxilliaryVariationalAutoencoder, LadderVariationalAutoencoder
    vae_kwargs = dict(
        x_dim=2,
        z_dim=2,
        h_dim=[64, 64]
    )
    warmup_epochs = 0
    device = get_device()
    seed = 0


@ex.automain
def run(device, vae_type, vae_kwargs, dataset, n_epochs, batch_size, learning_rate, importance_samples, warmup_epochs,
        seed):
    pprint(ex.current_run.config)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if dataset == 'spirals':
        train_dataset = Spirals(n_samples=1000, noise=0.05, rotation=0)
    elif dataset == 'moons':
        train_dataset = Moons(n_samples=1000, noise=0.05)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device=='cuda')
    
    # test1_dataset = Spirals(n_samples=1000, noise=0.05, rotation=np.pi/2)
    # test_loader = DataLoader(test1_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device=='cuda')

    model = getattr(vari.models.vae, vae_type)
    model = model(**vae_kwargs)
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    deterministic_warmup = DeterministicWarmup(n=warmup_epochs)

    epoch = 0
    i_update = 0
    best_elbo = -1e10
    try:
        while epoch < n_epochs:
            model.train()
            total_elbo, total_kl, total_log_px = 0, 0, 0
            beta = next(deterministic_warmup)
            for b, (x, _) in enumerate(train_loader):
                x = x.to(device)

                optimizer.zero_grad()
                
                # if epoch == 200 and b == 0:
                #     import IPython
                #     IPython.embed()

                x_iw = x.repeat(1, importance_samples).view(-1, x.shape[1])
                
                px, px_mu, px_sigma = model(x_iw)
                kl_divergence = model.kl_divergence

                likelihood = log_gaussian(x_iw, px_mu, px_sigma)
                elbo = likelihood #- beta * kl_divergence

                # Importance sampling
                # NOTE: Training is slow compared to TF version and ELBO values are muddy
                # NOTE: Results are identical with IW removed and IW=1
                # NOTE: Training without KL loss works but is slower and has lower final ELBO than TF
                # NOTE: 
                # TODO Are we mixing the order of the importance samples??
                elbo = log_sum_exp(elbo.view(-1, importance_samples, 1), axis=1, sum_op=torch.mean).view(-1, 1)  # (B, 1, 1)
                kl_divergence = log_sum_exp(kl_divergence.view(-1, importance_samples, 1), axis=1, sum_op=torch.mean).view(-1, 1)  # (B, 1, 1)
                likelihood = log_sum_exp(likelihood.view(-1, importance_samples, 1), axis=1, sum_op=torch.mean).view(-1, 1)  # (B, 1, 1)

                loss = - torch.mean(elbo)

                loss.backward()
                optimizer.step()

                total_elbo += elbo.mean().item()
                total_log_px += likelihood.mean().item()
                total_kl += kl_divergence.mean().item()

                i_update += 1
                ex.log_scalar(f'(batch) ELBO log p(x)', total_elbo / (b + 1), step=i_update)
                ex.log_scalar(f'(batch) log p(x|z)', total_log_px / (b + 1), step=i_update)
                ex.log_scalar(f'(batch) KL(q(z|x)||p(z))', total_kl / (b + 1), step=i_update)
                ex.log_scalar(f'(batch) ß * KL(q(z|x)||p(z))', beta * total_kl / (b + 1), step=i_update)
                

                # elbo_test = torch.Tensor([1,1,1,2,3,4,3,3,4])
                # log_sum_exp(elbo_test.view(-1, 3, 1), axis=1, sum_op=torch.mean).view(-1, 1)
                # elbo_test = torch.Tensor([1,2,3,4,5])
                # log_sum_exp(elbo_test.view(-1, 1, 1), axis=1, sum_op=torch.mean).view(-1, 1)
                
                # USE IW
                # torch.manual_seed(0)
                # importance_samples = 1
                # x_iw = x.repeat(1, importance_samples).view(-1, x.shape[1])
                # px, px_mu, px_sigma = model(x_iw)
                # kl_divergence = model.kl_divergence
                # likelihood = log_gaussian(x_iw, px_mu, px_sigma)
                # elbo = likelihood - beta * kl_divergence
                # print(elbo.view(-1, importance_samples, 1))
                # elbo = log_sum_exp(elbo.view(-1, importance_samples, 1), axis=1, sum_op=torch.mean).view(-1, 1)  # (B, 1, 1)
                # print(elbo)
                
                # # # DON'T USE IW
                # torch.manual_seed(0)
                # px, px_mu, px_sigma = model(x)
                # kl_divergence = model.kl_divergence
                # likelihood = log_gaussian(x, px_mu, px_sigma)
                # elbo = likelihood - beta * kl_divergence
                # print(elbo.view(-1, 1, 1))
                # elbo = log_sum_exp(elbo.view(-1, 1, 1), axis=1, sum_op=torch.mean).view(-1, 1)  # (B, 1, 1)
                # print(elbo)
                
            total_elbo = total_elbo / len(train_loader)
            total_log_px = total_log_px / len(train_loader)
            total_kl = total_kl / len(train_loader)
                
            ex.log_scalar(f'ELBO log p(x)', total_elbo, step=epoch)
            ex.log_scalar(f'log p(x|z)', total_log_px, step=epoch)
            ex.log_scalar(f'KL(q(z|x)||p(z))', total_kl, step=epoch)
            ex.log_scalar(f'ß * KL(q(z|x)||p(z))', beta * total_kl, step=epoch)

            print(f'Epoch {epoch} | ELBO {total_elbo:2.4f} | log p(x|z) {total_log_px:2.4f} | KL {total_kl:2.4f} | ß*KL {beta * total_kl:2.4f}')

            if epoch % 10 == 0 and total_elbo > best_elbo and deterministic_warmup.is_done:
                best_elbo = total_elbo
                torch.save(model.state_dict(), f'{ex.models_dir()}/model_state_dict.pkl')
                print(f'Saved model at ELBO {total_elbo:.4f}')
            
            epoch += 1
    except KeyboardInterrupt:
        print('Interrupted experiment')
