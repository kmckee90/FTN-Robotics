"""Shared behavior-cloning agent, support replay, evaluation, and checkpoints."""
from __future__ import annotations
import random
from dataclasses import asdict
import numpy as np
import torch
import torch.nn.functional as F
from models import LateralSmoother, anneal_ste_mask, kwta, make_actor, model_name, reactive_enabled
from models.search import ModelInference, frozen_actor

def disjoint_task_mask(ti: int, H: int, k: int) -> torch.Tensor:
    """k-hot mask for the fixed-mask baseline: task ti owns neurons
    [ti*k, (ti+1)*k). Caller must ensure (ti+1)*k <= H."""
    m = torch.zeros(H)
    m[ti * k : (ti + 1) * k] = 1.0
    return m

class BCAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg):
        if cfg.weight_decay != 0:
            raise ValueError('weight_decay must be zero to preserve inactive units')
        self.cfg = cfg
        self.act_dim = act_dim
        H, k = cfg.ftn_hidden_size, cfg.ftn_k
        self.actor = make_actor(obs_dim, act_dim, cfg)
        self.opt = torch.optim.Adam(
            (p for p in self.actor.parameters() if p.requires_grad), lr=cfg.bc_lr, weight_decay=0
        )
        if cfg.no_mask:
            self.m_a = torch.ones(H)
        elif cfg.fixed_mask:
            self.m_a = disjoint_task_mask(0, H, k)
        else:
            self.m_a = kwta(torch.randn(H), k=k)
        self.m_a_logits: torch.Tensor | None = None
        self.stored_m_a: dict[int, torch.Tensor] = {}
        # Lateral spatial-smoothing step applied to every reconfigured mask.
        self.smoother = LateralSmoother(
            H,
            k,
            square_dim=cfg.ftn_square_dim,
            kernel_size=cfg.lateral_kernel,
            lateral_steps=cfg.lateral_steps,
            lateral_polyak=cfg.lateral_polyak,
            lateral_noise_sd=cfg.lateral_noise_sd,
            torus=cfg.lateral_torus,
        )
        self.inference = ModelInference(self.actor, cfg, self.smoother)
        self.inference.state.masks[0].copy_(self.m_a)
        # Frozen global observation normalizer (set after data generation).
        self.obs_mean: torch.Tensor | None = None
        self.obs_std: torch.Tensor | None = None

    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor):
        self.obs_mean, self.obs_std = mean, std

    def _norm(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / self.obs_std

    def reset_optimizer_state(self):
        """Clear all weight-Adam moments and step counters after reconfiguration."""
        self.opt.state.clear()

    @torch.no_grad()
    def step_optimizer(self, masks):
        """Only update units used by this step's current/replay/segment masks.

        Replay membership and segment batches can change without a search. Clear
        inactive Adam moments on EVERY step as well as resetting after searches.
        """
        if any(group['weight_decay'] != 0 for group in self.opt.param_groups):
            raise ValueError('weight_decay must be zero to preserve inactive units')
        active = masks.detach().reshape(-1, self.actor.H).gt(.5).any(0)
        for parameter, axis in self.actor.unit_parameters():
            shape = [1] * parameter.ndim
            shape[axis] = self.actor.H
            inactive = (~active).reshape(shape)
            if parameter.grad is not None:
                parameter.grad.masked_fill_(inactive, 0)
            state = self.opt.state.get(parameter, {})
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in state:
                    state[key].masked_fill_(inactive, 0)
        self.opt.step()

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, mask_a: torch.Tensor | None = None) -> np.ndarray:
        m = self.m_a if mask_a is None else mask_a
        obs = self._norm(torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0))
        was_training = self.actor.training
        self.actor.eval()
        a = self.actor(obs, m).squeeze(0).cpu().numpy()
        if was_training:
            self.actor.train()
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def reconfig_actor_mask(
        self, obs: torch.Tensor, act: torch.Tensor, commit: bool,
        steps: int | None = None,
    ):
        """Search the k-hot actor mask that minimizes BC loss on the (already
        normalized) expert support batch using the configured model's inference.
        Noncommitting alternative-model searches are stateless cold evaluation.
        """
        cfg = self.cfg
        explicit_steps = steps
        steps = cfg.reconfig_steps if steps is None else steps
        if steps < 0:
            raise ValueError("Mask reconfiguration steps must be nonnegative")
        if model_name(cfg) != 'ftn' or reactive_enabled(cfg):
            masks, logits, stats = self.inference.search(
                obs.unsqueeze(0), act.unsqueeze(0), commit=commit,
                steps=steps if not commit else explicit_steps,
                stored=(torch.stack(list(self.stored_m_a.values())) if self.stored_m_a else None),
            )
            if commit:
                if (not reactive_enabled(cfg) or (stats['steps'] > 0).any()
                        or not torch.equal(self.m_a, masks[0])):
                    self.reset_optimizer_state()
                self.m_a, self.m_a_logits = masks[0], logits[0]
            return masks[0], {key: float(stats[key].mean()) for key in ('bc_init', 'bc_final')}
        H, k = cfg.ftn_hidden_size, cfg.ftn_k
        init = self.m_a_logits if cfg.mask_warm_start else None

        def bc_loss(soft_m):
            return F.mse_loss(self.actor(obs, soft_m), act)

        with frozen_actor(self.actor):
            m_new, logits_new, l_init, l_final = anneal_ste_mask(
                H,
                k,
                bc_loss,
                steps,
                cfg.reconfig_lr,
                init_logits=init,
                noise_scale=cfg.mask_init_noise_scale,
            )
        # Lateral spatial smoothing: cluster the searched mask into a contiguous
        # blob on the square grid (no-op when smoothing is disabled).
        m_new = self.smoother.smooth(m_new)
        if commit:
            self.reset_optimizer_state()
            self.m_a = m_new
            self.m_a_logits = logits_new
        return m_new, {"bc_init": l_init, "bc_final": l_final}

    def bc_update(self, obs: torch.Tensor, act: torch.Tensor, recalled: list):
        """One BC gradient step: MSE on the current batch (under m_a) summed
        with MSE on each recalled past chunk (under the mask it was trained
        under). Shared neurons accumulate gradients from both — the cross-task
        consolidation that keeps past tasks intact while learning the new one.
        `obs` tensors are already normalized. Returns the current-task loss."""
        terms = [(obs, act, self.m_a)] + recalled
        total = None
        cur_loss_val = 0.0
        for o, a, m in terms:
            loss = F.mse_loss(self.actor(o, m), a)
            if m is self.m_a:
                cur_loss_val = float(loss.item())
            total = loss if total is None else total + loss
        self.opt.zero_grad()
        total.backward()
        self.step_optimizer(torch.stack([m for _, _, m in terms]))
        self.inference.record_masks(self.m_a)
        return cur_loss_val

    def sample_recalled(self, archive):
        return archive.sample_filtered_multi(
            self.m_a,
            self.cfg.replay_max_chunks,
            self.cfg.replay_max_tuples,
            self.cfg.rho_low,
            self.cfg.rho_high,
        )

class SupportArchive:
    """Per-completed-task store of (normalized obs, act, m_a). Filtered sampling
    returns chunks whose mask overlaps the current mask within (rho_low,
    rho_high] — the consolidation targets. No-mask (identical masks, overlap=1)
    and fixed-mask (disjoint, overlap=0) never qualify, so consolidation is a
    no-op for those baselines."""

    def __init__(self):
        self.past: list[dict] = []

    def archive(self, obs: torch.Tensor, act: torch.Tensor, m_a: torch.Tensor):
        self.past.append(
            {
                "obs": obs.detach().clone(),
                "act": act.detach().clone(),
                "m_a": m_a.detach().clone(),
            }
        )

    @staticmethod
    def _overlap(m_cur: torch.Tensor, m_old: torch.Tensor) -> float:
        denom = float(m_cur.sum().clamp(min=1).item())
        return float((m_cur * m_old).sum().item()) / denom

    def sample_filtered_multi(self, m_a_cur, n_chunks, n_tuples, rho_low, rho_high):
        if not self.past:
            return []
        cands = [
            i
            for i, ph in enumerate(self.past)
            if rho_low < self._overlap(m_a_cur, ph["m_a"]) <= rho_high
        ]
        if not cands:
            return []
        chosen = np.random.choice(cands, size=min(n_chunks, len(cands)), replace=False)
        out = []
        for i in chosen:
            ph = self.past[int(i)]
            N = ph["obs"].shape[0]
            idx = torch.randint(0, N, (min(n_tuples, N),))
            out.append((ph["obs"][idx], ph["act"][idx], ph["m_a"]))
        return out

def make_support(demo: dict, agent: BCAgent, n_max: int):
    """Subsample up to n_max expert transitions → (normalized obs, act) tensors."""
    N = demo["obs"].shape[0]
    idx = np.random.choice(N, size=min(n_max, N), replace=False)
    obs = agent._norm(torch.tensor(demo["obs"][idx], dtype=torch.float32))
    act = torch.tensor(demo["act"][idx], dtype=torch.float32)
    return obs, act

def eval_task(
    env,
    agent: BCAgent,
    cfg,
    mask_a: torch.Tensor | None = None,
    reconfig_demo: dict | None = None,
    reconfig_every: int = 0,
) -> float:
    """Run cfg.eval_episodes greedy rollouts. With `reconfig_demo` (full FTN),
    recover task tj's mask from its expert support by a cold-started reconfig
    (commit=False): before episode 0, then again every `reconfig_every` episodes
    (reconfig_every=0 -> only once per cell, reused thereafter). Without
    `reconfig_demo`, roll out under the fixed `mask_a` (recall / no-mask)."""
    n_succ = 0
    for ep in range(cfg.eval_episodes):
        need_reconfig = reconfig_demo is not None and (
            ep == 0 or (reconfig_every >= 1 and ep % reconfig_every == 0)
        )
        if need_reconfig:
            s_obs, s_act = make_support(reconfig_demo, agent, cfg.reconfig_n)
            saved = agent.m_a_logits
            agent.m_a_logits = None  # cold-start this reconfig
            mask_a, _ = agent.reconfig_actor_mask(
                s_obs, s_act, commit=False,
                steps=getattr(cfg, "eval_reconfig_steps", None),
            )
            agent.m_a_logits = saved
        obs, _ = env.reset()
        for _ in range(cfg.eval_max_steps):
            a = agent.act(obs, mask_a=mask_a)
            obs, _, term, trunc, info = env.step(a)
            if float(info.get("success", 0.0)) > 0.5:
                n_succ += 1
                break
            if term or trunc:
                break
    return n_succ / cfg.eval_episodes

def eval_prior_task(env, agent, cfg, demo, task_index):
    """Measure the untrained actor without changing weights or training RNGs."""
    py_state, np_state = random.getstate(), np.random.get_state()
    env_rng, logits = env.rng, agent.m_a_logits
    training = agent.actor.training
    params = list(agent.actor.parameters())
    flags = [p.requires_grad for p in params]
    try:
        with torch.random.fork_rng():
            seed = cfg.seed + 100000 + task_index
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
                mask = disjoint_task_mask(task_index, cfg.ftn_hidden_size, cfg.ftn_k)
                return eval_task(env, agent, cfg, mask_a=mask)
            # Stored-mask mode has no learned masks yet: search for the prior.
            return eval_task(env, agent, cfg, reconfig_demo=demo,
                             reconfig_every=cfg.reconfig_eval)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        env.rng, agent.m_a_logits = env_rng, logits
        agent.actor.train(training)
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)

def save_checkpoint(
    path: str,
    agent: BCAgent,
    cfg,
    obs_dim: int,
    act_dim: int,
    phases_done: int,
):
    """Save the FTN model to a single file (overwritten after each task). Holds
    the actor weights, the frozen obs normalizer, every per-task mask learned so
    far, and enough config to rebuild the agent for eval/visualization."""
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "inference": agent.inference.state_dict(),
            "m_a_logits": None if agent.m_a_logits is None else agent.m_a_logits.detach().cpu(),
            "mask_archive": checkpoint_masks(agent, cfg),
            "obs_mean": None
            if agent.obs_mean is None
            else agent.obs_mean.detach().cpu(),
            "obs_std": None if agent.obs_std is None else agent.obs_std.detach().cpu(),
            "m_a": agent.m_a.detach().cpu(),
            "stored_m_a": {i: m.detach().cpu() for i, m in agent.stored_m_a.items()},
            "cfg": asdict(cfg),
            "tasks": list(cfg.tasks),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "phases_done": phases_done,
        },
        path,
    )
    print(f"  [ckpt] saved model with {phases_done} task(s) trained -> {path}")


def checkpoint_masks(agent, cfg):
    """All final learned masks: one per task, or last-visit mask per segment."""
    if hasattr(agent, "mask_archive"):
        return agent.mask_archive
    ids = sorted(agent.stored_m_a)
    if not ids:
        return None
    masks = np.stack([agent.stored_m_a[i].detach().cpu().numpy() > 0 for i in ids])
    return {
        "packed": torch.from_numpy(np.packbits(masks, axis=1)),
        "task_indices": torch.tensor(ids, device="cpu"),
        "hidden_size": cfg.ftn_hidden_size,
        "scope": "committed mask for each completed training task",
    }
