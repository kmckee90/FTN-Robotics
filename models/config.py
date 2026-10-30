"""Architecture and inference options shared by both experiment dataclasses."""
from dataclasses import dataclass
import math


@dataclass
class ModelConfig:
    model: str = 'ftn'  # ftn, lora, bayesftn, bayeslora
    # Shared architecture, mask geometry and ordinary inference defaults.
    ftn_hidden_size: int = 2304
    ftn_k: int = 256
    ftn_inner_size: int = 8
    ftn_layers: int = 8
    ftn_dropout_p: float = 0.0
    ftn_square_dim: int = 48
    lateral_steps: int = 0
    lateral_polyak: float = 1.0
    lateral_kernel: int = 3
    lateral_noise_sd: float = 0.0
    lateral_torus: bool = False
    reconfig_steps: int = 30
    reconfig_lr: float = 0.3
    mask_init_noise_scale: float = 1e-5
    mask_warm_start: bool = False
    no_mask: bool = False
    fixed_mask: bool = False
    stored_mask: bool = False
    # Cross-task support replay; MT10 explicitly disables this mechanism.
    consolidate: bool = True
    replay_max_chunks: int = 4
    replay_max_tuples: int = 1024
    rho_low: float = 0.2
    rho_high: float = 0.8
    lora_layers: int = 4
    lora_width: int = 64
    lora_rank: int = 4
    lora_alpha: float = 16.0
    lora_weight_scale: float = 1.0
    lora_b0_scale: float = 0.01
    lora_dropout: float = 0.0
    lora_bias: bool = True
    lora_reconfig_algorithm: str = 'annealste'  # or fast (sigmoid relaxation)
    lora_reconfig_optimizer: str = 'Adam'
    lora_warm_start_reconfig: bool = False
    lora_reactive_continuous_updates: bool = False
    lora_changepoint_reconfig_steps: int = 30
    lora_match_stored: bool = False
    lora_match_stored_after: int = 0
    lora_match_stored_topk: int = 1
    reconfig_noise_scale: float = 1.0
    reconfig_overlap_mode: str = 'bump'  # bump, gated, seen_logits
    reconfig_overlap_lambda: float = 0.0
    reconfig_overlap_threshold: float = 0.5
    reconfig_overlap_temperature: float = 0.2
    # None selects reactive detection for Bayesian models, scheduled for others.
    reconfig_reactive: bool | None = None
    reconfig_reactive_sd: float = 5.0
    reconfig_reactive_ema: float = 0.95
    bayes_reconfig_steps: int = 3
    bayes_reconfig_lr: float = 0.3
    bayes_reconfig_sample_size: int | None = None
    bayes_reconfig_take_best: bool = False
    bayes_kl_weight: float = 1.0
    bayes_temperature: float = 1.0
    bayes_prior_decay: float = 1.0
    bayes_stochastic_mask: bool = True
    bayes_sample_type: str = 'gumbel'  # or annealste
    bayes_changepoint_reconfig_steps: int = 30
    bayes_changepoint_reconfig_lr: float = 0.3
    bayes_changepoint_reconfig_sample_size: int | None = None
    bayes_changepoint_kl_weight: float = 0.0
    bayes_changepoint_temperature: float = 1.0
    bayes_changepoint_prior_decay: float = 0.0
    bayes_changepoint_stochastic_mask: bool = True
    eval_reconfig_lr: float = 0.3
    eval_reconfig_sample_size: int | None = None
    eval_reconfig_temperature: float = 1.0
    eval_reconfig_restarts: int = 1
    eval_reconfig_take_best: bool = False


def model_name(cfg):
    name = getattr(cfg, 'model', 'ftn').lower().replace('_', '').replace('-', '')
    return {'base': 'ftn', 'bayes': 'bayesftn'}.get(name, name)


def is_bayesian(cfg):
    return model_name(cfg) in ('bayesftn', 'bayeslora')


def reactive_enabled(cfg):
    value = getattr(cfg, 'reconfig_reactive', None)
    return is_bayesian(cfg) if value is None else value


def with_model_defaults(cfg):
    # Legacy checkpoints have no model options and remain base FTN checkpoints.
    for key, value in vars(ModelConfig()).items():
        if not hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def validate_model_config(cfg):
    with_model_defaults(cfg)
    if model_name(cfg) not in ('ftn', 'lora', 'bayesftn', 'bayeslora'):
        raise ValueError('model must be ftn, lora, bayesftn, or bayeslora')
    for key in ('lora_width', 'lora_rank', 'lora_changepoint_reconfig_steps',
                'lora_match_stored_topk', 'bayes_reconfig_steps',
                'bayes_changepoint_reconfig_steps', 'eval_reconfig_restarts'):
        if getattr(cfg, key) < 1:
            raise ValueError(f'{key} must be positive')
    for key in ('lora_layers', 'lora_match_stored_after'):
        if getattr(cfg, key) < 0:
            raise ValueError(f'{key} must be nonnegative')
    for key in ('bayes_reconfig_sample_size', 'bayes_changepoint_reconfig_sample_size',
                'eval_reconfig_sample_size'):
        if getattr(cfg, key) is not None and getattr(cfg, key) < 1:
            raise ValueError(f'{key} must be positive or None')
    for key in ('lora_alpha', 'lora_b0_scale', 'bayes_temperature',
                'bayes_changepoint_temperature', 'bayes_changepoint_reconfig_lr',
                'eval_reconfig_lr', 'eval_reconfig_temperature', 'reconfig_overlap_temperature'):
        if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('lora_weight_scale', 'bayes_reconfig_lr', 'bayes_kl_weight',
                'bayes_changepoint_kl_weight', 'reconfig_noise_scale',
                'reconfig_overlap_lambda', 'reconfig_reactive_sd'):
        if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    for key in ('bayes_prior_decay', 'bayes_changepoint_prior_decay', 'reconfig_overlap_threshold'):
        if not 0 <= getattr(cfg, key) <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    for key in ('lora_dropout', 'reconfig_reactive_ema'):
        if not 0 <= getattr(cfg, key) < 1:
            raise ValueError(f'{key} must be in [0, 1)')
    choices = {'lora_reconfig_algorithm': ('annealste', 'fast'),
               'lora_reconfig_optimizer': ('Adam', 'SGD', 'RMSprop'),
               'bayes_sample_type': ('gumbel', 'annealste'),
               'reconfig_overlap_mode': ('bump', 'gated', 'seen_logits')}
    for key, values in choices.items():
        if getattr(cfg, key) not in values:
            raise ValueError(f'{key} must be one of {values}')
