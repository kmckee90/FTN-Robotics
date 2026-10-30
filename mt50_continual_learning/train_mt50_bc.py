"""Sequential MT50 behavior cloning with prior + per-task success matrix."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataclasses import dataclass, asdict
import json
import os
import random
import time
import numpy as np
import torch
from models import ModelConfig, reactive_enabled
from runtime import run_cli, validate_config, configure_runtime
import csv
import metaworld
from bc import BCAgent, SupportArchive, disjoint_task_mask, eval_task, eval_prior_task, make_support, save_checkpoint
from data import DEFAULT_DATA_DIR, StandardTaskEnv, fit_normalizer, generate_or_load_demos
from visualization import plot_matrix

DEFAULT_TASKS = ('reach-v3', 'push-v3', 'door-open-v3', 'drawer-open-v3', 'drawer-close-v3', 'button-press-topdown-v3', 'window-open-v3', 'window-close-v3') + tuple(sorted(set(metaworld.ALL_V3_ENVIRONMENTS) - set(('reach-v3', 'push-v3', 'door-open-v3', 'drawer-open-v3', 'drawer-close-v3', 'button-press-topdown-v3', 'window-open-v3', 'window-close-v3'))))

@dataclass
class ExpertBCConfig(ModelConfig):

    # Tasks and execution
    tasks: tuple = DEFAULT_TASKS
    num_tasks: int | None = None  # Use tasks[:num_tasks]; None keeps all configured tasks.
    seed: int = 0
    device: str = "auto"  # CUDA when available, otherwise CPU
    num_threads: int = 3
    output_path: str = ""
    data_dir: str = DEFAULT_DATA_DIR
    log_every: int = 5

    # Scripted-expert demonstrations
    n_demos: int = 250
    demo_max_steps: int = 400
    demo_tail: int = 100

    # Behavior cloning
    bc_epochs: int = 30
    bc_batch_size: int = 256
    bc_lr: float = 0.0003
    weight_decay: float = 0.0
    eval_episodes: int = 4
    eval_max_steps: int = 500

    # Checkpoints
    ckpt_path: str = ""
    kt_dir: str = "checkpoints"

    # AnnealedSTE mask search
    eval_reconfig_steps: int | None = 60
    reconfig_n: int = 4096
    reconfig_train: int = 10
    reconfig_eval: int = 0

    # Artifact destination
    artifact_root: str = str(Path(__file__).resolve().parent / "artifacts")

def smoke_config() -> ExpertBCConfig:
    return ExpertBCConfig(
        tasks=("drawer-open-v3", "drawer-close-v3"),
        n_demos=5,
        bc_epochs=5,
        eval_episodes=3,
        eval_max_steps=200,
        ftn_hidden_size=64,
        ftn_k=16,
        ftn_inner_size=4,
        ftn_layers=2,
        ftn_square_dim=8,  # 8**2 = 64 = ftn_hidden_size (one block)
        reconfig_steps=10,
        reconfig_n=128,
    )

def build_envs(task_names, seed):
    benchmark = metaworld.MT50(seed=seed)
    return [StandardTaskEnv(name, benchmark.train_classes[name],
                           [t for t in benchmark.train_tasks if t.env_name == name],
                           seed + i + 1)
            for i, name in enumerate(task_names)]

def evaluate_cell(env, agent, cfg, demo, ti, tj):
    # Use the same evaluation seeds in every row, without perturbing training RNGs.
    py_state, np_state = random.getstate(), np.random.get_state()
    env_rng, logits = env.rng, agent.m_a_logits
    training = agent.actor.training
    params = list(agent.actor.parameters())
    grad_flags = [p.requires_grad for p in params]
    try:
        with torch.random.fork_rng():
            seed = cfg.seed + 100000 + tj
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.rng = np.random.default_rng(seed)
            agent.actor.eval()
            for p in params:
                p.requires_grad_(False)
            if cfg.no_mask:
                return eval_task(env, agent, cfg, mask_a=agent.m_a)
            if cfg.fixed_mask:
                mask = disjoint_task_mask(tj, cfg.ftn_hidden_size, cfg.ftn_k)
                return eval_task(env, agent, cfg, mask_a=mask)
            if cfg.stored_mask and tj <= ti:
                return eval_task(env, agent, cfg, mask_a=agent.stored_m_a[tj])
            return eval_task(env, agent, cfg, reconfig_demo=demo,
                             reconfig_every=cfg.reconfig_eval)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        env.rng, agent.m_a_logits = env_rng, logits
        agent.actor.train(training)
        for p, flag in zip(params, grad_flags):
            p.requires_grad_(flag)

def save_progress(matrix, cfg, phases_done, reports=False):
    path = Path(cfg.output_path)
    row_labels = ["prior", *cfg.tasks]
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, matrix=matrix, tasks=np.array(cfg.tasks),
             row_labels=np.array(row_labels),
             phases_done=phases_done, cells_done=np.isfinite(matrix).sum())
    os.replace(tmp, path)
    if not reports:
        return
    csv_path = path.with_suffix(".csv")
    tmp = csv_path.with_suffix(".tmp.csv")
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trained_through / eval_task", *cfg.tasks])
        for name, row in zip(row_labels, matrix):
            writer.writerow([name, *["" if np.isnan(v) else f"{v:.4f}" for v in row]])
    os.replace(tmp, csv_path)
    lines = [f"MT50 BC: {phases_done}/{len(cfg.tasks)} tasks completed",
             "Columns: " + " | ".join(f"{i+1}:{t}" for i, t in enumerate(cfg.tasks)),
             "Row 0: untrained prior. Remaining rows: recall below / transfer above the task diagonal."]
    for i, (name, row) in enumerate(zip(row_labels, matrix)):
        values = " ".join("   -" if np.isnan(v) else f"{v:.2f}" for v in row)
        lines.append(f"{i:02d} {name:30s} {values}")
    txt_path = path.with_suffix(".txt")
    tmp = txt_path.with_suffix(".tmp.txt")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, txt_path)
    plot_matrix(matrix, cfg.tasks, path.with_suffix(".png"), phases_done)

def train(cfg: ExpertBCConfig):
    validate_config(cfg)
    configure_runtime(cfg)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if len(set(cfg.tasks)) != len(cfg.tasks) or not cfg.tasks:
        raise ValueError("Tasks must be nonempty and unique")
    unknown = set(cfg.tasks) - set(DEFAULT_TASKS)
    if unknown:
        raise ValueError(f"Unknown MT50 tasks: {sorted(unknown)}")
    if cfg.fixed_mask and len(cfg.tasks) * cfg.ftn_k > cfg.ftn_hidden_size:
        raise ValueError("fixed-mask requires hidden-size >= number of tasks * k")
    for field in ("n_demos", "demo_max_steps", "bc_epochs", "bc_batch_size",
                  "eval_episodes", "eval_max_steps", "reconfig_n", "log_every"):
        if getattr(cfg, field) < 1:
            raise ValueError(f"{field} must be positive")
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    if cfg.ckpt_path:
        Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_path).with_suffix(".config.json").write_text(
        json.dumps(asdict(cfg), indent=2))
    envs = build_envs(cfg.tasks, cfg.seed)
    try:
        return _train_envs(cfg, envs)
    finally:
        for env in envs:
            env.env.close()

def _train_envs(cfg, envs):
    N = len(envs)
    obs_dim, act_dim = envs[0].obs_dim, envs[0].act_dim

    if cfg.fixed_mask and N * cfg.ftn_k > cfg.ftn_hidden_size:
        raise ValueError(
            f"fixed_mask needs N*k <= H for disjoint subsets, got "
            f"N={N}, k={cfg.ftn_k}, H={cfg.ftn_hidden_size}"
        )

    print("Loading shared expert demonstrations...")
    demos = generate_or_load_demos(envs, cfg)
    obs_mean, obs_std = fit_normalizer(demos)

    agent = BCAgent(obs_dim, act_dim, cfg)
    agent.set_normalizer(obs_mean, obs_std)
    archive = SupportArchive()

    no_reconfig = cfg.fixed_mask or cfg.no_mask
    baseline_label = (
        "no-mask" if cfg.no_mask else ("fixed-mask" if cfg.fixed_mask else "")
    )

    matrix = np.full((N + 1, N), np.nan, dtype=np.float32)
    save_progress(matrix, cfg, 0, reports=True)
    t_start = time.time()

    print("\n=== Prior: evaluation before any task training ===", flush=True)
    for tj, prior_env in enumerate(envs):
        matrix[0, tj] = eval_prior_task(prior_env, agent, cfg, demos[prior_env.name], tj)
        print(f"  [prior] {prior_env.name:25s} sr={matrix[0, tj]:.2f}", flush=True)
        save_progress(matrix, cfg, 0)
    save_progress(matrix, cfg, 0, reports=True)

    kt_base = os.path.splitext(os.path.basename(cfg.output_path))[0]
    if cfg.kt_dir:
        os.makedirs(cfg.kt_dir, exist_ok=True)

    def save_kt_checkpoint(n_trained: int):
        kt_path = os.path.join(cfg.kt_dir, f"{kt_base}_ntasks{n_trained}.pt")
        save_checkpoint(kt_path, agent, cfg, obs_dim, act_dim, n_trained)

    for ti, env in enumerate(envs):
        print(f"\n=== Phase {ti}/{N - 1}: {env.name} ===")
        # Export the model before this phase (ntasks0 is the untrained prior).
        if cfg.kt_dir:
            save_kt_checkpoint(ti)
        agent.reset_optimizer_state()

        if cfg.fixed_mask:
            agent.m_a = disjoint_task_mask(ti, cfg.ftn_hidden_size, cfg.ftn_k)
            print(f"  fixed-mask: neurons [{ti * cfg.ftn_k}, {(ti + 1) * cfg.ftn_k})")

        demo = demos[env.name]
        obs_full = agent._norm(torch.tensor(demo["obs"], dtype=torch.float32))
        act_full = torch.tensor(demo["act"], dtype=torch.float32)
        n_trans = obs_full.shape[0]

        # 1) Mask reconfig on the expert support (FTN / stored-mask only) ----
        if not no_reconfig and not reactive_enabled(cfg):
            s_obs, s_act = make_support(demo, agent, cfg.reconfig_n)
            _, stats = agent.reconfig_actor_mask(s_obs, s_act, commit=True)
            print(f"  reconfig: BC {stats['bc_init']:.4f} -> {stats['bc_final']:.4f}")
        elif not no_reconfig:
            print("  reactive inference: update on minibatches without task-boundary resets")
        else:
            print(f"  {baseline_label} baseline: skipping reconfig")
        agent.stored_m_a[ti] = agent.m_a.detach().clone()

        # 2) Behavior cloning under the committed mask ----------------------
        self_bs = cfg.bc_batch_size
        batch_ctr = 0
        for epoch in range(cfg.bc_epochs):
            perm = torch.randperm(n_trans)
            ep_loss, n_b = 0.0, 0
            for start in range(0, n_trans, self_bs):
                bidx = perm[start : start + self_bs]
                batch_ctr += 1
                # Every reconfig_train'th minibatch, re-search the mask on this
                # minibatch (warm-started from the running logits) before the
                # step. reconfig_train=0 disables it (mask frozen post phase-start).
                if (
                    not no_reconfig
                    and (reactive_enabled(cfg) or (
                        cfg.reconfig_train >= 1 and batch_ctr % cfg.reconfig_train == 0))
                ):
                    agent.reconfig_actor_mask(
                        obs_full[bidx], act_full[bidx], commit=True
                    )
                recalled = (
                    agent.sample_recalled(archive)
                    if (cfg.consolidate and not no_reconfig)
                    else []
                )
                loss = agent.bc_update(obs_full[bidx], act_full[bidx], recalled)
                ep_loss += loss
                n_b += 1
            if (epoch + 1) % cfg.log_every == 0 or epoch == cfg.bc_epochs - 1:
                elapsed = time.time() - t_start
                print(
                    f"  epoch {epoch + 1:>3d}/{cfg.bc_epochs}  "
                    f"bc_loss={ep_loss / max(1, n_b):.4f}  ({elapsed:.0f}s)"
                )

        agent.stored_m_a[ti] = agent.m_a.detach().clone()
        # 3) Archive this phase's support for cross-task consolidation -------
        if not no_reconfig:
            arch_obs, arch_act = make_support(demo, agent, cfg.replay_max_tuples * 8)
            archive.archive(arch_obs, arch_act, agent.m_a)

        # 4) Eval every task -------------------------------------------------
        print(f"  --- Eval after phase {ti} ---")
        for tj in range(N):
            tag = "<-" if tj == ti else "  "
            sr = evaluate_cell(envs[tj], agent, cfg, demos[envs[tj].name], ti, tj)
            matrix[ti + 1, tj] = sr
            print(f"    [{tag}] {envs[tj].name:25s} sr={sr:.2f}")

            save_progress(matrix, cfg, ti + int(tj == N - 1))

        save_progress(matrix, cfg, ti + 1, reports=True)
        row = " ".join(f"{v:.2f}" for v in matrix[ti + 1])
        print(f"  [ROW {ti + 1}/{N}] {env.name}: {row}", flush=True)
        print(f"  Live matrix: {Path(cfg.output_path).with_suffix('.png')}", flush=True)

        # 5) Checkpoint the model (single file, overwritten each phase) ---------
        if cfg.ckpt_path:
            save_checkpoint(cfg.ckpt_path, agent, cfg, obs_dim, act_dim, ti + 1)

    # Final snapshot (all N tasks trained).
    if cfg.kt_dir:
        save_kt_checkpoint(N)

    print("\n=== Eval matrix (rows: training phase, cols: eval task) ===")
    for ti, label in enumerate(["prior", *cfg.tasks]):
        row = " ".join(
            "  — " if np.isnan(matrix[ti, tj]) else f"{matrix[ti, tj]:.2f}"
            for tj in range(N)
        )
        print(f"  step {ti} ({label}): {row}")

    return matrix

def main():
    run_cli(ExpertBCConfig, smoke_config, train, "mt50")

if __name__ == "__main__":
    main()
