"""FTN and LoRA architectures, with ordinary or Bayesian mask inference."""
from .ftn import (FTNActor, ParallelMLP, LateralSmoother, anneal_ste_mask,
                  anneal_ste_mask_batched, kwta, mask_to_grid, mask_to_vec)
from .lora import LoRAActor, LoRALinear
from .config import ModelConfig, is_bayesian, model_name, reactive_enabled, with_model_defaults


class BayesFTN(FTNActor):
    """FTN backbone selected with persistent Bayesian inference in ModelInference."""


class BayesLoRA(LoRAActor):
    """LoRA backbone selected with persistent Bayesian inference in ModelInference."""


def make_actor(obs_dim, act_dim, cfg):
    with_model_defaults(cfg)
    name = model_name(cfg)
    if name in ('lora', 'bayeslora'):
        return (BayesLoRA if name == 'bayeslora' else LoRAActor)(obs_dim, act_dim, cfg)
    if name not in ('ftn', 'bayesftn'):
        raise ValueError(f'Unknown model {cfg.model!r}')
    cls = BayesFTN if name == 'bayesftn' else FTNActor
    return cls(obs_dim, act_dim, cfg.ftn_hidden_size, cfg.ftn_layers,
               cfg.ftn_inner_size, cfg.ftn_dropout_p)
