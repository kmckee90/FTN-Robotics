"""MetaWorld environments and automatically cached scripted-expert demonstrations."""
from __future__ import annotations
import os
import metaworld
import numpy as np
import torch
from metaworld.policies import ENV_POLICY_MAP

class StandardTaskEnv:
    """One MT10 env class + its 50 goal variants. Standard reset (random goal
    variant, default initial state) — no cheat init."""

    def __init__(self, name: str, env_cls, tasks, seed: int):
        self.name = name
        self.env_cls = env_cls
        self.env = env_cls()
        self.tasks = tasks
        self.rng = np.random.default_rng(seed)
        self.env.set_task(self.tasks[0])
        obs, _ = self.env.reset()
        self.obs_dim = int(obs.shape[0])
        self.act_dim = int(self.env.action_space.shape[0])

    def reset(self):
        task = self.tasks[self.rng.integers(len(self.tasks))]
        self.env.set_task(task)
        obs, info = self.env.reset()
        return obs.astype(np.float32), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action.astype(np.float32))
        return obs.astype(np.float32), float(r), bool(term), bool(trunc), info

def build_envs(task_names, seed: int) -> list[StandardTaskEnv]:
    mt10 = metaworld.MT10(seed=seed)
    out = []
    for i, name in enumerate(task_names):
        tasks = [t for t in mt10.train_tasks if t.env_name == name]
        out.append(StandardTaskEnv(name, mt10.train_classes[name], tasks, seed + i + 1))
    return out

def _rollout_expert(env: StandardTaskEnv, policy, cfg):
    """One scripted-expert episode. Returns (obs[T,D], act[T,A], success) where
    T is truncated to `first_success + demo_tail` (full horizon if no success)."""
    obs_buf, act_buf = [], []
    obs, _ = env.reset()
    first_succ = -1
    for t in range(cfg.demo_max_steps):
        a = np.clip(policy.get_action(obs), -1.0, 1.0).astype(np.float32)
        obs_buf.append(obs)
        act_buf.append(a)
        obs, _, term, trunc, info = env.step(a)
        if float(info.get("success", 0.0)) > 0.5 and first_succ < 0:
            first_succ = t
        if term or trunc:
            break
    success = int(first_succ >= 0)
    cut = (
        len(obs_buf)
        if first_succ < 0
        else min(len(obs_buf), first_succ + 1 + cfg.demo_tail)
    )
    return (
        np.asarray(obs_buf[:cut], dtype=np.float32),
        np.asarray(act_buf[:cut], dtype=np.float32),
        success,
    )

def generate_or_load_demos(envs, cfg) -> dict[str, dict]:
    """Per task: a cached dict {obs:[N,D], act:[N,A]} of concatenated expert
    transitions. Cache key includes the demo-generation params + seed so the
    four baselines reuse identical data."""
    os.makedirs(cfg.data_dir, exist_ok=True)
    demos: dict[str, dict] = {}
    for env in envs:
        key = f"{env.name}_n{cfg.n_demos}_s{cfg.seed}_t{cfg.demo_tail}_m{cfg.demo_max_steps}.npz"
        path = os.path.join(cfg.data_dir, key)
        if os.path.exists(path):
            d = np.load(path)
            demos[env.name] = {"obs": d["obs"], "act": d["act"]}
            print(f"  [cache] {env.name:25s} {d['obs'].shape[0]} transitions")
            continue
        policy = ENV_POLICY_MAP[env.name]()
        obs_all, act_all, n_succ = [], [], 0
        for _ in range(cfg.n_demos):
            o, a, s = _rollout_expert(env, policy, cfg)
            obs_all.append(o)
            act_all.append(a)
            n_succ += s
        obs_cat = np.concatenate(obs_all, axis=0)
        act_cat = np.concatenate(act_all, axis=0)
        np.savez(path, obs=obs_cat, act=act_cat)
        demos[env.name] = {"obs": obs_cat, "act": act_cat}
        print(
            f"  [gen]   {env.name:25s} {obs_cat.shape[0]} transitions "
            f"({n_succ}/{cfg.n_demos} expert demos succeeded)"
        )
    return demos

def fit_normalizer(demos: dict[str, dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Global observation standardizer over all tasks' expert data (frozen,
    shared across every task and condition so comparisons are fair). Actions
    are already in [-1, 1] and left unscaled."""
    all_obs = np.concatenate([d["obs"] for d in demos.values()], axis=0)
    mean = torch.tensor(all_obs.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(all_obs.std(axis=0), dtype=torch.float32).clamp_min(1e-6)
    return mean, std

def _split_by_lengths(obs_cat, act_cat, lengths):
    """Inverse of the concatenate-with-lengths on-disk layout."""
    trajs, i = [], 0
    for L in lengths:
        L = int(L)
        trajs.append((obs_cat[i : i + L], act_cat[i : i + L]))
        i += L
    return trajs

def generate_or_load_trajectories(envs, cfg) -> dict[str, list]:
    """Per task: a list of (obs[T,D], act[T,A]) expert trajectories with their
    per-episode boundaries preserved (segments need contiguous single-episode
    windows). Cached as concatenated arrays plus a `lengths` index; the cache
    key mirrors the flat demo cache params so the same rollouts are
    reused, just kept un-flattened."""
    os.makedirs(cfg.data_dir, exist_ok=True)
    trajs: dict[str, list] = {}
    for env in envs:
        key = (
            f"traj_{env.name}_n{cfg.n_demos}_s{cfg.seed}_t{cfg.demo_tail}"
            f"_m{cfg.demo_max_steps}.npz"
        )
        path = os.path.join(cfg.data_dir, key)
        if os.path.exists(path):
            d = np.load(path)
            trajs[env.name] = _split_by_lengths(d["obs"], d["act"], d["lengths"])
            n = int(d["lengths"].sum())
            print(
                f"  [cache] {env.name:25s} {len(d['lengths'])} trajectories "
                f"({n} transitions)"
            )
            continue
        policy = ENV_POLICY_MAP[env.name]()
        obs_list, act_list, lengths, n_succ = [], [], [], 0
        for _ in range(cfg.n_demos):
            o, a, s = _rollout_expert(env, policy, cfg)
            obs_list.append(o)
            act_list.append(a)
            lengths.append(o.shape[0])
            n_succ += s
        obs_cat = np.concatenate(obs_list, axis=0)
        act_cat = np.concatenate(act_list, axis=0)
        lengths = np.asarray(lengths, dtype=np.int64)
        np.savez(path, obs=obs_cat, act=act_cat, lengths=lengths)
        trajs[env.name] = list(zip(obs_list, act_list))
        print(
            f"  [gen]   {env.name:25s} {len(obs_list)} trajectories "
            f"({obs_cat.shape[0]} transitions, "
            f"{n_succ}/{cfg.n_demos} expert demos succeeded)"
        )
    return trajs

def segment_windows(T: int, seg_len: int, stride: int) -> list[tuple[int, int]]:
    """(start, end) index pairs of contiguous windows of length `seg_len` over a
    length-T trajectory, advancing by `stride`. The last window is snapped to
    end at T so the trajectory tail is always covered. Short trajectories
    (T <= seg_len) yield a single full-length window."""
    if T <= seg_len:
        return [(0, T)]
    stride = max(1, stride)
    starts = list(range(0, T - seg_len + 1, stride))
    if starts[-1] != T - seg_len:
        starts.append(T - seg_len)
    return [(s, s + seg_len) for s in starts]
