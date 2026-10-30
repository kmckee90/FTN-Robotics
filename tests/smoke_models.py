"""Explicit pod integration check using real shared expert data and MuJoCo.

Run: python tests/smoke_models.py --output /home/dev/ftn_model_validation
Uses tiny models/budgets; this checks execution, not learning quality.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import random
import numpy as np
import torch
from mt50_continual_learning import train_mt50_bc as mt50
from mt10_segment_representation import train_mt10_transitions_bc_par as mt10
from rendering import load_agent, RenderConfig
from runtime import configure_runtime, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/home/dev/ftn_model_validation')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--models', nargs='+', choices=('ftn', 'lora', 'bayesftn', 'bayeslora'),
                        default=['ftn', 'lora', 'bayesftn', 'bayeslora'])
    parser.add_argument('--experiments', nargs='+', choices=('mt50', 'mt10'), default=['mt50', 'mt10'])
    args = parser.parse_args()
    tasks = ('drawer-open-v3', 'drawer-close-v3')
    print('Building real MetaWorld environments once for the integration run', flush=True)
    envs = mt50.build_envs(tasks, 0)
    try:
        for model in args.models:
            for experiment in args.experiments:
                folder = Path(args.output) / model / experiment
                folder.mkdir(parents=True, exist_ok=True)
                cfg = (mt50.ExpertBCConfig if experiment == 'mt50' else mt10.ParSegmentBCConfig)(
                    model=model, device=args.device, tasks=tasks, n_demos=2,
                    ftn_hidden_size=16, ftn_k=4, ftn_square_dim=4,
                    ftn_layers=1, ftn_inner_size=3, num_threads=1,
                    lora_layers=1, lora_width=8, lora_rank=2,
                    reconfig_steps=2, bayes_reconfig_steps=2,
                    bayes_changepoint_reconfig_steps=3, log_every=1)
                if experiment == 'mt50':
                    cfg.bc_epochs, cfg.bc_batch_size = 1, 128
                    cfg.eval_episodes, cfg.eval_max_steps, cfg.eval_reconfig_steps = 1, 5, 2
                    cfg.reconfig_n, cfg.reconfig_train = 8, 1
                    cfg.kt_dir, cfg.output_path = '', str(folder / 'results.npz')
                    cfg.ckpt_path = str(folder / 'model.pt')
                    checkpoint = cfg.ckpt_path
                else:
                    cfg.segment_length, cfg.segment_stride = 16, 64
                    cfg.segment_batch, cfg.segment_cycles, cfg.segment_epochs = 4, 2, 2
                    cfg.segment_model_dir = str(folder)
                    cfg.batch_mode = 'random'  # posterior must follow segment IDs
                    checkpoint = str(folder / cfg.ckpt_name)
                validate_config(cfg, segments=experiment == 'mt10')
                configure_runtime(cfg)
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                print(f'=== {model} / {experiment} ===', flush=True)
                (mt50 if experiment == 'mt50' else mt10)._train_envs(cfg, envs)
                loaded, _, saved = load_agent(checkpoint, RenderConfig(device=args.device, num_threads=1))
                obs, _ = envs[0].reset()
                action = loaded.act(obs)
                assert action.shape == (4,) and np.isfinite(action).all()
                assert saved['mask_archive'] is not None
                # Exercise the visualization mask-recovery path without rendering.
                from rendering import observe_expert
                options = RenderConfig(device=args.device, max_steps=3, reconfig_steps=2)
                recovered = observe_expert(envs[0], envs[0].tasks[0], 0, loaded, options, 1)
                assert recovered['masks'].shape == (3, 16)
                print(f'PASS {model} / {experiment}: train, eval, checkpoint, action, mask recovery', flush=True)
    finally:
        for env in envs:
            env.env.close()


if __name__ == '__main__':
    main()
