"""Parallel segment behavior cloning with recurrent batched mask searches."""
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
from models import ModelConfig, model_name, reactive_enabled
from runtime import run_cli, validate_config, configure_runtime
from models import anneal_ste_mask_batched
from bc import BCAgent, save_checkpoint
from data import DEFAULT_DATA_DIR, build_envs, fit_normalizer, generate_or_load_trajectories, segment_windows
from visualization import plot_losses

DEFAULT_TASKS = ('reach-v3', 'push-v3', 'door-open-v3', 'drawer-open-v3', 'drawer-close-v3', 'button-press-topdown-v3', 'window-open-v3', 'window-close-v3', 'peg-insert-side-v3', 'pick-place-v3')

@dataclass
class ParSegmentBCConfig(ModelConfig):

    # Tasks and execution
    tasks: tuple = DEFAULT_TASKS
    seed: int = 0
    device: str = "auto"  # CUDA when available, otherwise CPU
    num_threads: int = 3
    data_dir: str = DEFAULT_DATA_DIR
    log_every: int = 5

    # Scripted-expert demonstrations
    n_demos: int = 150
    demo_max_steps: int = 400
    demo_tail: int = 100

    # Behavior cloning
    bc_lr: float = 0.0003
    weight_decay: float = 0.0

    # Legacy experiment defaults when no model YAML is supplied.
    ftn_hidden_size: int = 1024
    ftn_k: int = 64

    # Spatial mask smoothing
    ftn_square_dim: int = 32
    lateral_polyak: float = 0.9
    lateral_kernel: int = 5

    # Segment warm starts
    mask_warm_start: bool = True

    # Parallel segment training
    segment_length: int = 4
    segment_stride: int = 1
    segment_bc_steps: int = 1
    segment_epochs: int = 2
    segment_model_dir: str = ""
    ckpt_name: str = "model.pt"
    segment_batch: int = 512
    segment_cycles: int = 10
    batch_mode: str = 'contiguous'

    # Artifact destination
    artifact_root: str = str(Path(__file__).resolve().parent / "artifacts")

def smoke_config() -> ParSegmentBCConfig:
    return ParSegmentBCConfig(
        tasks=("drawer-open-v3",),
        n_demos=3,
        ftn_hidden_size=64,
        ftn_k=16,
        ftn_inner_size=4,
        ftn_layers=2,
        ftn_square_dim=8,  # 8**2 = 64 = ftn_hidden_size (one block)
        reconfig_steps=10,
        segment_length=16,
        segment_stride=8,
        segment_batch=8,
        segment_cycles=4,
        segment_bc_steps=1,
        segment_epochs=2,
    )

def batched_actor(actor, obs_b: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """FTNActor forward for a *batch of segments each under its own mask*.

    obs_b : (S, L, D) — S segments of length L.
    masks : (S, H)    — one k-hot (or STE soft) mask per segment.
    returns (S, L, A) actions.

    Dispatches to the backbone's batched implementation: FTN gates precomputed
    features; LoRA applies each segment mask at every frozen-backbone layer."""
    return actor.forward_batched(obs_b, masks)

def batched_reconfig(
    agent: BCAgent,
    obs_b: torch.Tensor,
    act_b: torch.Tensor,
    cfg: ParSegmentBCConfig,
    init_logits: torch.Tensor | None,
    segment_ids=None,
):
    """Configured inference: recover an independent k-hot mask for each
    of the S segments in the batch, against the current actor weights. Returns
    (masks (S,H) hard+smoothed, logits (S,H), bc_init, bc_final).

    The actor weights are frozen for the search (only the S mask-logit vectors
    are optimized), so no actor gradients are computed here."""
    H, k, S = cfg.ftn_hidden_size, cfg.ftn_k, obs_b.shape[0]
    if model_name(cfg) != 'ftn' or reactive_enabled(cfg):
        if segment_ids is None:
            # Direct calls represent a stable batch; training always passes IDs.
            if agent.inference.segment_state is None:
                agent.inference.init_segments(S)
            segment_ids = list(range(S))
        masks, logits, stats = agent.inference.search(
            obs_b, act_b, commit=True, ids=segment_ids, init_logits=init_logits,
            stored=(torch.stack(list(agent.stored_m_a.values())) if agent.stored_m_a else None),
        )
        if not reactive_enabled(cfg) or (stats['steps'] > 0).any():
            agent.reset_optimizer_state()
        return masks, logits, float(stats['bc_init'].mean()), float(stats['bc_final'].mean())

    def loss_fn(soft_masks):  # soft_masks: (S, H) -> per-segment loss (S,)
        y = batched_actor(agent.actor, obs_b, soft_masks)
        return ((y - act_b) ** 2).mean(dim=(1, 2))

    # Freeze actor params so the search graph doesn't waste work on their grads.
    flags = [p.requires_grad for p in agent.actor.parameters()]
    for p in agent.actor.parameters():
        p.requires_grad_(False)
    try:
        masks, logits, l0, l1 = anneal_ste_mask_batched(
            H,
            k,
            loss_fn,
            cfg.reconfig_steps,
            cfg.reconfig_lr,
            n_search=S,
            init_logits=init_logits,
            noise_scale=cfg.mask_init_noise_scale,
        )
    finally:
        for p, flag in zip(agent.actor.parameters(), flags):
            p.requires_grad_(flag)

    # Lateral spatial smoothing, per segment (no-op when smoothing is disabled).
    if agent.smoother.enabled:
        masks = torch.stack([agent.smoother.smooth(m) for m in masks])
    agent.reset_optimizer_state()
    return masks, logits, l0, l1

def batched_bc_step(agent: BCAgent, obs_b, act_b, masks) -> float:
    """One BC gradient step summed-then-averaged over the S segments, each under
    its own committed mask. Overlapping neurons receive gradient from every
    segment that uses them — the co-training signal that lets the batch organize
    a joint subspace structure over the cycles."""
    y = batched_actor(agent.actor, obs_b, masks)
    loss = ((y - act_b) ** 2).mean(dim=(1, 2)).mean()  # mean over segments
    agent.opt.zero_grad()
    loss.backward()
    agent.step_optimizer(masks)
    agent.inference.record_masks(masks)
    return float(loss.item())

def build_batches(segments, batch_size: int, mode: str) -> list[list[int]]:
    """Group segment indices into parallel batches for one epoch.

    random     — permute all segments, then chunk (batch membership varies each
                 call: pass reshuffles).
    trajectory — one batch per trajectory (chunked if a trajectory has > batch
                 windows); never mixes two trajectories. Batch order shuffled.
    contiguous — chunk the natural window order into fixed-size batches (may
                 straddle trajectory boundaries). Batch order shuffled.

    For trajectory/contiguous the membership is deterministic; only the visiting
    order is randomized so weight updates don't see the same order every epoch."""
    n = len(segments)
    if mode == "random":
        order = np.random.permutation(n)
        return [order[i : i + batch_size].tolist() for i in range(0, n, batch_size)]
    if mode == "contiguous":
        batches = [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]
    elif mode == "trajectory":
        groups: dict[int, list[int]] = {}
        for idx, (tid, _s, _e) in enumerate(segments):
            groups.setdefault(tid, []).append(idx)
        batches = [
            members[i : i + batch_size]
            for members in groups.values()
            for i in range(0, len(members), batch_size)
        ]
    else:
        raise ValueError(f"unknown batch_mode {mode!r}")
    np.random.shuffle(batches)  # randomize visiting order, keep membership
    return batches

def _n_batches(segments, batch_size: int, mode: str) -> int:
    """Batch count per epoch (analytic, so it doesn't consume the RNG)."""
    if mode in ("random", "contiguous"):
        return (len(segments) + batch_size - 1) // batch_size
    counts: dict[int, int] = {}
    for tid, _s, _e in segments:
        counts[tid] = counts.get(tid, 0) + 1
    return sum((c + batch_size - 1) // batch_size for c in counts.values())

def train_par_segments(cfg: ParSegmentBCConfig):
    validate_config(cfg, segments=True)
    configure_runtime(cfg)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    envs = build_envs(cfg.tasks, cfg.seed)
    try:
        return _train_envs(cfg, envs)
    finally:
        for env in envs:
            env.env.close()

def _train_envs(cfg, envs):
    obs_dim, act_dim = envs[0].obs_dim, envs[0].act_dim

    print("Loading shared expert trajectories...")
    trajs = generate_or_load_trajectories(envs, cfg)

    flat = {
        name: {
            "obs": np.concatenate([o for o, _ in trajs[name]], axis=0),
            "act": np.concatenate([a for _, a in trajs[name]], axis=0),
        }
        for name in trajs
    }
    obs_mean, obs_std = fit_normalizer(flat)

    agent = BCAgent(obs_dim, act_dim, cfg)
    agent.set_normalizer(obs_mean, obs_std)

    # Precompute normalized trajectory tensors; index only full-length windows so
    # a batch stacks into a clean (S, L, D). Short-trajectory windows (T < L) are
    # dropped and counted (batching needs a uniform length).
    L = cfg.segment_length
    traj_tensors: list[tuple[torch.Tensor, torch.Tensor]] = []
    segments: list[tuple[int, int, int]] = []
    per_task_counts: dict[str, int] = {}
    n_dropped = 0
    for name in cfg.tasks:
        n_seg_task = 0
        for o, a in trajs[name]:
            obs_t = agent._norm(torch.tensor(o, dtype=torch.float32))
            act_t = torch.tensor(a, dtype=torch.float32)
            tid = len(traj_tensors)
            traj_tensors.append((obs_t, act_t))
            for s, e in segment_windows(o.shape[0], L, cfg.segment_stride):
                if e - s != L:
                    n_dropped += 1
                    continue
                segments.append((tid, s, e))
                n_seg_task += 1
        per_task_counts[name] = n_seg_task

    n_seg = len(segments)
    if n_seg == 0:
        raise SystemExit(
            f"No full-length ({L}) segments — every trajectory is shorter than "
            f"segment_length. Lower --segment-length."
        )
    overlap = max(0, L - cfg.segment_stride)
    n_batches_per_epoch = _n_batches(segments, cfg.segment_batch, cfg.batch_mode)
    print(
        f"\n{n_seg} segments (len={L}, stride={cfg.segment_stride}, "
        f"overlap<={overlap}) across {len(cfg.tasks)} task(s):"
    )
    for name in cfg.tasks:
        print(f"    {name:25s} {per_task_counts[name]} segments")
    if n_dropped:
        print(f"    (dropped {n_dropped} short window(s) < len {L})")
    print(
        f"Parallel: batch={cfg.segment_batch} segments ({cfg.batch_mode}), "
        f"{cfg.segment_cycles} reconfig<->step cycle(s)/batch, "
        f"{cfg.segment_bc_steps} BC step(s)/cycle, "
        f"{'warm' if cfg.mask_warm_start else 'cold'}-start; "
        f"{n_batches_per_epoch} batches x {cfg.segment_epochs} epoch(s)."
    )

    def gather(bidx):
        obs_b = torch.stack([traj_tensors[segments[i][0]][0][segments[i][1]:segments[i][2]] for i in bidx])
        act_b = torch.stack([traj_tensors[segments[i][0]][1][segments[i][1]:segments[i][2]] for i in bidx])
        return obs_b, act_b

    if model_name(cfg) != 'ftn' or reactive_enabled(cfg):
        agent.inference.init_segments(n_seg)

    # Keep every segment's last committed mask, bit-packed on CPU for export.
    learned_masks = np.zeros((n_seg, (cfg.ftn_hidden_size + 7) // 8), dtype=np.uint8)
    losses = []
    t_start = time.time()
    for epoch in range(cfg.segment_epochs):
        batches = build_batches(segments, cfg.segment_batch, cfg.batch_mode)
        ep_loss, n_upd = 0.0, 0
        for bi, bidx in enumerate(batches, start=1):
            obs_b, act_b = gather(bidx)

            logits = None  # cold start the first cycle regardless
            batch_loss, l0, l1 = 0.0, 0.0, 0.0
            for _ in range(cfg.segment_cycles):
                init = logits if cfg.mask_warm_start else None
                masks, logits, l0, l1 = batched_reconfig(agent, obs_b, act_b, cfg, init, segment_ids=bidx)
                for _ in range(cfg.segment_bc_steps):
                    batch_loss = batched_bc_step(agent, obs_b, act_b, masks)
            if epoch == cfg.segment_epochs - 1:
                learned_masks[bidx] = np.packbits(masks.detach().cpu().numpy() > 0, axis=1)
            ep_loss += batch_loss
            n_upd += 1

            if bi % cfg.log_every == 0 or bi == len(batches):
                elapsed = time.time() - t_start
                print(
                    f"  epoch {epoch + 1}/{cfg.segment_epochs}  "
                    f"batch {bi:>4d}/{len(batches)}  "
                    f"bc_loss={ep_loss / max(1, n_upd):.4f}  "
                    f"(reconfig {l0:.4f}->{l1:.4f})  ({elapsed:.0f}s)"
                )

        losses.append(ep_loss / max(1, n_upd))

    os.makedirs(cfg.segment_model_dir, exist_ok=True)
    np.savez(Path(cfg.segment_model_dir) / "results.npz", loss=np.array(losses), tasks=np.array(cfg.tasks), segment_counts=np.array([per_task_counts[n] for n in cfg.tasks]))
    plot_losses(losses, Path(cfg.segment_model_dir) / "loss.png")
    ckpt_path = os.path.join(cfg.segment_model_dir, cfg.ckpt_name)
    agent.mask_archive = {
        "packed": torch.from_numpy(learned_masks),
        "task_indices": torch.from_numpy(np.repeat(np.arange(len(cfg.tasks)),
            [per_task_counts[name] for name in cfg.tasks])),
        "hidden_size": cfg.ftn_hidden_size,
        "scope": "last committed mask for every training segment (final epoch)",
    }
    save_checkpoint(ckpt_path, agent, cfg, obs_dim, act_dim, len(cfg.tasks))

    config_path = os.path.join(
        cfg.segment_model_dir, cfg.ckpt_name.replace(".pt", ".config.json")
    )
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    print(f"Saved config to {config_path}")
    return ckpt_path

def main():
    run_cli(ParSegmentBCConfig, smoke_config, train_par_segments, "mt10")

if __name__ == "__main__":
    main()
