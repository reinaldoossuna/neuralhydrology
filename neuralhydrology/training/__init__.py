import logging
from functools import partial
import warnings
from typing import List

import torch

import neuralhydrology.training.loss as loss
from neuralhydrology.training import regularization
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.dummy_lr_scheduler import DummyLRS

LOGGER = logging.getLogger(__name__)


def get_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Get specific optimizer object, depending on the run configuration.

    Currently only 'Adam' and 'AdamW' are supported.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be optimized.
    cfg : Config
        The run configuration.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer object that can be used for model training.
    """
    match cfg.optimizer.lower():
        case "adam":
            return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate[0])
        case "adamw":
            return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate[0])
        case "radam":
            return torch.optim.RAdam(model.parameters(), lr=cfg.learning_rate[0])
        case _:
            raise NotImplementedError(
                f"{cfg.optimizer} not implemented or not linked in `get_optimizer()`"
            )


def get_loss_obj(cfg: Config) -> loss.BaseLoss:
    """Get loss object, depending on the run configuration.

    Currently supported are 'MSE', 'NSE', 'RMSE', 'GMMLoss', 'CMALLoss', and 'UMALLoss'.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    loss.BaseLoss
        A new loss instance that implements the loss specified in the config or, if different, the loss required by the
        head.
    """
    if cfg.loss.lower() == "mse":
        loss_obj = loss.MaskedMSELoss(cfg)
    elif cfg.loss.lower() == "nse":
        loss_obj = loss.MaskedNSELoss(cfg)
    elif cfg.loss.lower() == "weightednse":
        warnings.warn(
            "'WeightedNSE loss has been removed. Use 'NSE' with 'target_loss_weights'",
            FutureWarning,
        )
        loss_obj = loss.MaskedNSELoss(cfg)
    elif cfg.loss.lower() == "rmse":
        loss_obj = loss.MaskedRMSELoss(cfg)
    elif cfg.loss.lower() == "gmmloss":
        loss_obj = loss.MaskedGMMLoss(cfg)
    elif cfg.loss.lower() == "cmalloss":
        loss_obj = loss.MaskedCMALLoss(cfg)
    elif cfg.loss.lower() == "umalloss":
        loss_obj = loss.MaskedUMALLoss(cfg)
    else:
        raise NotImplementedError(f"{cfg.loss} not implemented or not linked in `get_loss()`")

    return loss_obj


def get_regularization_obj(cfg: Config) -> List[regularization.BaseRegularization]:
    """Get list of regularization objects.

    Currently, only the 'tie_frequencies' regularization is implemented.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    List[regularization.BaseRegularization]
        List of regularization objects that will be added to the loss during training.
    """
    regularization_modules = []
    for reg_item in cfg.regularization:
        if isinstance(reg_item, str):
            reg_name = reg_item
            reg_weight = 1.0
        else:
            reg_name, reg_weight = reg_item
        if reg_name == "tie_frequencies":
            regularization_modules.append(
                regularization.TiedFrequencyMSERegularization(cfg=cfg, weight=reg_weight)
            )
        elif reg_name == "forecast_overlap":
            regularization_modules.append(
                regularization.ForecastOverlapMSERegularization(cfg=cfg, weight=reg_weight)
            )
        else:
            raise NotImplementedError(
                f"{reg_name} not implemented or not linked in `get_regularization_obj()`."
            )

    return regularization_modules


def get_lr_scheduler(cfg: Config) -> torch.optim.lr_scheduler.LRScheduler:
    """Get specific lr_scheduler object, depending on the run configuration.

    Currently only 'multiplicative' and 'multistep' are supported.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler
        Scheduler object that can be used for adjusts the learning rate during optimization.
    """
    match cfg.lr_scheduler["type"].lower():
        case "multiplicative":
            return partial(
                torch.optim.lr_scheduler.MultiplicativeLR,
                lr_lambda=lambda epoch: cfg.lr_scheduler["rate"],
            )
        case "multistep":
            return partial(
                torch.optim.lr_scheduler.MultiStepLR,
                milestones=cfg.lr_scheduler["milestones"],
                gamma=cfg.lr_scheduler["gamma"],
            )
        case _:
            return DummyLRS
