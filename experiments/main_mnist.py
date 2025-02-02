import datetime

from collections import defaultdict

import numpy as np
import logging
import pickle
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from pprint import pprint
from sacred import SETTINGS

import vari.models.vae
import vari.datasets

from model_utils.experiment import Experiment
from vari.layers import GaussianLayer
from vari.models import build_dense_vae, build_conv_vae, build_conv_dense_vae
from vari.models.vae import get_copy_latents
from vari.utilities import get_device, summary
from vari.inference import DeterministicWarmup, FreeNatsCooldown

import IPython


SETTINGS.CONFIG.READ_ONLY_CONFIG = False  # Allow modifications to variables defined in config
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
    importance_samples = 1
    learning_rate = 3e-4
    warmup_epochs = 0
    free_nats = None
    
    max_epochs_without_improvement = 500
    epoch_checkpoint_interval = 50
    
    model_type = 'conv'
    if True:  #model_type == 'dense':
        build_kwargs = dict(
            x_dim=784, #3*32*32,  #784,
            z_dim=[5, 2],
            h_dim=[[512, 512], [256, 256]],
            activation=torch.nn.LeakyReLU(),
            batchnorm=False,
        )
    elif model_type == 'conv':
        encoders = [
            {
                'name': 'Conv2dSequentialCoder',
                'kwargs': {
                    'in_channels': 1,
                    'n_filters': [128, 256],
                    'kernels': [(4, 4), (4, 4)],
                    'strides': [2, 2],
                    'transpose': False,
                    'activation': torch.nn.LeakyReLU
                },
            },
            # {
            #     'name': 'DenseSequentialCoder',
            #     'kwargs': {
            #         'x_dim': 64,
            #         'h_dim': [128, 128],
            #         'activation': torch.nn.LeakyReLU
            #     }
            # }
        ]
        encoder_distributions = [
            {
                'name': 'GaussianLayer',
                'kwargs': {
                    'in_features': 256,
                    'out_features': 64,
                }
            },
            # {
            #     'name': 'GaussianLayer',
            #     'kwargs': {
            #         'in_features': 128,
            #         'out_features': 32,
            #     }
            # }
        ]
        decoders = [
            # {
            #     'name': 'DenseSequentialCoder',
            #     'kwargs': {
            #         'x_dim': 32,
            #         'h_dim': [128, 128],
            #         'activation': torch.nn.LeakyReLU
            #     }
            # },
            {
                'name': 'Conv2dSequentialCoder',
                'kwargs': {
                    'in_channels': 64,
                    'n_filters': [128, 128],
                    'kernels': [(4, 4), (4, 4)],
                    'strides': [2, 2],
                    'transpose': True,
                    'activation': torch.nn.LeakyReLU()
                },
            },
        ]
        decoder_distributions = [
            # {
            #     'name': 'GaussianLayer',
            #     'kwargs': {
            #         'in_features': 256,
            #         'out_features': 64,
            #     }
            # },
            {
                'name': 'BernoulliLayer',
                'kwargs': {
                    'dimensions': 2,
                }
            }
        ]
        build_kwargs = dict(
            encoders=encoders,
            decoders=decoders,
            encoder_distributions=encoder_distributions,
            decoder_distributions=decoder_distributions,
        )
        model, model_kwargs = build_conv_vae(**build_kwargs)
    else:
        raise ValueError(f'Unknown model_type {model_type}')
    
    num_cpu_workers = 2
    test_importance_samples = set([1, importance_samples, 100])
    device = get_device()
    seed = 0

    build_kwargs['encoder_distribution'] = ['GaussianLayer'] * len(build_kwargs['z_dim'])
    build_kwargs['decoder_distribution'] = ['GaussianLayer'] * (len(build_kwargs['z_dim']) - 1)
    if 'Continuous' in dataset_name:
        # build_kwargs['decoder_distribution'].append('GaussianLayer')
        # build_kwargs['decoder_distribution'].append('ContinuousBernoulliLayer')
        build_kwargs['decoder_distribution'].append('BetaLayer')
    elif 'Binarized' in dataset_name:
        build_kwargs['decoder_distribution'].append('BernoulliLayer')
        
    tag_day = datetime.datetime.now().day
    tag_month = datetime.datetime.now().month
    tag_year = datetime.datetime.now().year


def loss_is_naninf(loss):
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        if torch.isnan(loss).any():
            ex.logger.warning(f'Epoch {epoch:3d} | Batch {b:3d} | The loss was NaN! (skipped)')
        else:
            ex.logger.warning(f'Epoch {epoch:3d} | Batch {b:3d} | The loss was Inf! (skipped)')
        return True
    return False


@ex.automain
def run(device, dataset_name, dataset_kwargs, build_kwargs, n_epochs, batch_size, learning_rate, importance_samples,
        warmup_epochs, free_nats, max_epochs_without_improvement, epoch_checkpoint_interval, test_importance_samples,
        num_cpu_workers, seed):

    # Print config, set threads and seed
    pprint(ex.current_run.config)
    if device == 'cpu':
        torch.set_num_threads(num_cpu_workers)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get dataset, model, optimizer and other
    dataset = getattr(vari.datasets, dataset_name)
    train_dataset = dataset(split='train', **dataset_kwargs)
    test_dataset = dataset(split='test', **dataset_kwargs)
    print(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_cpu_workers, pin_memory=device.type=='cuda')
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=num_cpu_workers, pin_memory=device.type=='cuda')

    model, model_kwargs = build_dense_vae(**build_kwargs)
    torch.save(model_kwargs, f'{ex.models_dir()}/model_kwargs.pkl')
    model.to(device)
    print(model)
    summary(model, model.in_shape, batch_size=batch_size * importance_samples)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    deterministic_warmup = DeterministicWarmup(n=warmup_epochs)
    free_nats_cooldown = FreeNatsCooldown(constant_epochs=int(warmup_epochs * 3), cooldown_epochs=200,
                                          start_val=free_nats, end_val=None)

    epoch = 0
    i_update = 0
    epoch_last_checkpoint = 0
    best_elbo = -1e10
    pz_samples = model.encoder[-1].distribution.prior.sample(torch.Size([1000]))

    try:
        while epoch < n_epochs:
            model.train()
            total_elbo, total_kl, total_likelihood, total_kls = 0, 0, 0, defaultdict(lambda: 0)
            total_elbo_1iw, total_kl_1iw, total_likelihood_1iw, total_kls_1iw = 0, 0, 0, defaultdict(lambda: 0)
            beta = next(deterministic_warmup)
            free_nats = next(free_nats_cooldown)
            for b, (x, _) in enumerate(train_loader):
                x = x.to(device, non_blocking=True)

                optimizer.zero_grad()

                x = x.view(x.shape[0], *model.in_shape)
                elbo, likelihood, kl_divergence = model.elbo(x, importance_samples=importance_samples, beta=beta,
                                                                free_nats=free_nats, reduce_importance_samples=False)
                kl_divergences_1iw = {k: v[0] for k, v in model.kl_divergences.items()}
                elbo_1iw, likelihood_1iw, kl_divergence_1iw = elbo[0], likelihood[0], kl_divergence[0]
                kl_divergences = model.kl_divergences
                elbo, likelihood, kl_divergence = model.reduce_importance_samples(elbo, likelihood, kl_divergences, importance_samples)

                loss = - torch.mean(elbo)
                if loss_is_naninf(loss):
                    continue

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
            if free_nats is not None:
                ex.log_scalar(f'Free nats', free_nats, step=epoch)
                
            s = f'Epoch {epoch:3d} | ELBO {total_elbo: 2.4f} | log p(x|z) {total_likelihood: 2.4f} | KL {total_kl:2.4f} | ß*KL {beta * total_kl:2.4f}'
            if len(total_kls) > 1:
                for k, v in total_kls.items():
                    s += f' | KL {k} {v / len(train_loader):2.4f}'
            print(s)

            if epoch % epoch_checkpoint_interval == 0:
                # Evaluate model
                model.eval()
                for iws in sorted(test_importance_samples):
                    total_elbo, total_kl, total_likelihood, total_kls = 0, 0, 0, defaultdict(lambda: 0)
                    # Dynamic batch size depending on the number of importance samples
                    test_loader = DataLoader(test_dataset, batch_size=(batch_size * 3) // iws + 1, shuffle=True,
                                             num_workers=num_cpu_workers, pin_memory=device=='cuda')
                    for b, (x, _) in enumerate(test_loader):
                        x = x.to(device)

                        x = x.view(x.shape[0], *model.in_shape)
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
                    x = x.view(x.shape[0], *model.in_shape)
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
                    epoch_last_checkpoint = epoch

                if epoch - epoch_last_checkpoint >= max_epochs_without_improvement and deterministic_warmup.is_done:
                    break  # End training loop

            epoch += 1

    except KeyboardInterrupt:
        print('Interrupted experiment')
        return f'ELBO={best_elbo:.2f}, p(x|z)={best_likelihood:.2f}, KL(q||p)={best_kl:.2f}'

    return f'ELBO={best_elbo:.2f}, p(x|z)={best_likelihood:.2f}, KL(q||p)={best_kl:.2f}'
