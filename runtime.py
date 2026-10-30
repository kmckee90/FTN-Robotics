"""CLI overrides, artifact routing, and validation shared by both experiments."""
import argparse
from dataclasses import fields
from pathlib import Path
import json
import os
from dataclasses import asdict


DEFAULT_RUN_NAMES = {"mt10": "mt10-segments", "mt50": "mt50-continual"}


def named_run(artifact_root, name):
    if not name or name in (".", "..") or Path(name).name != name or "\\" in name:
        raise ValueError("--name must be a single nonempty folder name")
    return Path(artifact_root).expanduser().resolve() / name


def validate_config(cfg, segments=False):
    from models.config import validate_model_config
    validate_model_config(cfg)
    if cfg.weight_decay != 0:
        raise ValueError('weight_decay must be zero to preserve inactive units')
    if not segments and getattr(cfg, 'num_tasks', None) is not None:
        if cfg.num_tasks < 1:
            raise ValueError("num_tasks must be positive")
        cfg.tasks = cfg.tasks[:cfg.num_tasks]
    if not cfg.tasks or len(set(cfg.tasks)) != len(cfg.tasks):
        raise ValueError("tasks must be nonempty and unique")
    for name in ('n_demos', 'demo_max_steps', 'log_every', 'ftn_hidden_size',
                 'ftn_inner_size', 'ftn_layers'):
        if getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 1 <= cfg.ftn_k <= cfg.ftn_hidden_size:
        raise ValueError("ftn_k must be between 1 and ftn_hidden_size")
    if cfg.ftn_square_dim < 1 or cfg.ftn_hidden_size % cfg.ftn_square_dim**2:
        raise ValueError("ftn_hidden_size must be divisible by ftn_square_dim squared")
    if cfg.num_threads < 1:
        raise ValueError('num_threads must be positive')
    if cfg.lateral_kernel < 1 or cfg.lateral_kernel % 2 == 0:
        raise ValueError('lateral_kernel must be positive and odd')
    if cfg.reconfig_steps < 0 or cfg.demo_tail < 0:
        raise ValueError("reconfig_steps and demo_tail must be nonnegative")
    if cfg.bc_lr <= 0 or cfg.reconfig_lr <= 0:
        raise ValueError("learning rates must be positive")
    if segments:
        # Segment co-training has no archive-based cross-task support replay.
        # This experiment rule intentionally takes precedence over model presets.
        cfg.consolidate = False
        cfg.replay_max_chunks = 0
        for name in ('segment_length', 'segment_stride', 'segment_batch',
                     'segment_cycles', 'segment_bc_steps', 'segment_epochs'):
            if getattr(cfg, name) < 1:
                raise ValueError(f"{name} must be positive")
        if cfg.no_mask or cfg.fixed_mask:
            raise ValueError("Parallel segment training requires searched masks")
        if cfg.batch_mode not in ('random', 'trajectory', 'contiguous'):
            raise ValueError("batch_mode must be random, trajectory, or contiguous")
    else:
        if sum((cfg.no_mask, cfg.fixed_mask, cfg.stored_mask)) > 1:
            raise ValueError("Choose only one mask baseline")
        if cfg.eval_reconfig_steps is not None and cfg.eval_reconfig_steps < 0:
            raise ValueError("eval_reconfig_steps must be nonnegative")


def run_cli(config_type, smoke_config, train, experiment):
    defaults = config_type()
    parser = argparse.ArgumentParser(description=train.__module__)
    parser.add_argument('--smoke', action='store_true', help='Use the small configuration at the top of the script.')
    parser.add_argument('--name', default=None, help='Run folder under this experiment\'s artifacts directory.')
    parser.add_argument('--model-config', default=None, metavar='NAME_OR_PATH',
                        help='Filename under models/presets/ (.yaml optional), or explicit path. '
                             'Precedence: defaults < YAML < smoke overrides < CLI.')
    # Dataclass defaults are the single source of configuration, including CLI defaults.
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        flag = '--' + field.name.replace('_', '-')
        kwargs = {'default': argparse.SUPPRESS, 'help': f'Default: {value}'}
        if field.name == 'no_mask':
            kwargs['action'] = 'store_true'
            parser.add_argument('--masked', dest='no_mask', action='store_false', default=argparse.SUPPRESS)
        elif isinstance(value, bool) or field.name == 'reconfig_reactive':
            kwargs['action'] = argparse.BooleanOptionalAction
        elif isinstance(value, tuple):
            kwargs.update(nargs='+', type=str)
        else:
            kwargs['type'] = type(value) if value is not None else int
        parser.add_argument(flag, **kwargs)
    args = vars(parser.parse_args())
    smoke = args.pop('smoke')
    name = args.pop('name')
    model_config = args.pop('model_config')
    if name is None:
        name = DEFAULT_RUN_NAMES[experiment] + ('-smoke' if smoke else '')
    try:
        named_run(defaults.artifact_root, name)
    except ValueError as exc:
        parser.error(str(exc))
    cfg = config_type()
    model_source = None
    if model_config:
        from models.preset_config import load_model_yaml
        try:
            overrides, model_source = load_model_yaml(model_config)
        except ValueError as exc:
            parser.error(str(exc))
        for key, value in overrides.items():
            setattr(cfg, key, value)
    if smoke:
        # Only the explicitly changed smoke settings override the selected model.
        smoke_defaults = smoke_config()
        for field in fields(defaults):
            value = getattr(smoke_defaults, field.name)
            if value != getattr(defaults, field.name):
                setattr(cfg, field.name, value)
    for key, value in args.items():
        setattr(cfg, key, tuple(value) if key == 'tasks' else value)
    run = named_run(cfg.artifact_root, name)
    if experiment == 'mt50':
        cfg.output_path = cfg.output_path or str(run / 'results.npz')
        cfg.ckpt_path = cfg.ckpt_path or str(run / 'model.pt')
        if cfg.kt_dir == 'checkpoints':
            cfg.kt_dir = str(run / 'checkpoints')
        if Path(cfg.output_path).suffix != '.npz':
            parser.error('--output-path must end in .npz')
    else:
        cfg.segment_model_dir = cfg.segment_model_dir or str(run)
        if Path(cfg.ckpt_name).name != cfg.ckpt_name or not cfg.ckpt_name.endswith('.pt'):
            parser.error('--ckpt-name must be a .pt filename')
    try:
        validate_config(cfg, segments=experiment == 'mt10')
    except ValueError as exc:
        parser.error(str(exc))
    run.mkdir(parents=True, exist_ok=True)
    if model_source is not None:
        (run / 'model_config.yaml').write_text(model_source)
    (run / 'config.json').write_text(json.dumps(asdict(cfg), indent=2) + '\n')
    checkpoint = Path(cfg.ckpt_path if experiment == 'mt50' else Path(cfg.segment_model_dir) / cfg.ckpt_name).resolve()
    (run / 'run.json').write_text(json.dumps({
        'name': name, 'experiment': experiment,
        'checkpoint': os.path.relpath(checkpoint, run),
        'visualizations': 'visualizations',
        'model_config': 'model_config.yaml' if model_source is not None else None,
    }, indent=2) + '\n')
    print(f'Artifacts: {run}', flush=True)
    train(cfg)


def configure_runtime(cfg):
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    torch.set_num_threads(cfg.num_threads)
    torch.set_default_device(device)
    print(f"Training device: {device}", flush=True)
