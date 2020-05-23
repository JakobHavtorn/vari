import logging
import os

import torch

import vari.settings


LOGGER = logging.getLogger()


def load_models(run_ids, model_class=vari.models.HierarchicalVariationalAutoencoder, device=torch.device('cpu'),
                experiments_dir=vari.settings.EXPERIMENTS_DIR):
    models = dict()
    for model_abbr, run_id in run_ids.items():
        run_id = str(run_id)

        # Load the saved kwargs
        kwargs = torch.load(
            os.path.join(experiments_dir, run_id, 'model_kwargs.pkl'),
            map_location=device,
        )
        model = model_class(**kwargs).to(device)

        try:
            model.load_state_dict(torch.load(
                os.path.join(experiments_dir, run_id, 'model_state_dict.pkl'),
                map_location=device
            ))
            models[model_abbr] = model
            LOGGER.info(f'Loaded ID {run_id} {model_abbr.upper()} on {device}')
        except Exception as exc:
            LOGGER.error(f'Failed loading {model_abbr.upper()} ID {run_id}')
            LOGGER.error(exc)
    return models


def save_model(model, path):
    raise NotImplementedError
