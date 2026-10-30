"""MetaWorld environments and shared, episode-indexed expert demonstrations."""
from __future__ import annotations
from pathlib import Path
import metaworld
import numpy as np
import torch

DEFAULT_DATA_DIR = "/home/dev/ftn_data/mt50_experts"
DATA_VERSION = 1

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

def dataset_path(cfg, name):
    root = Path(cfg.data_dir).expanduser().resolve()
    workspace = Path(__file__).resolve().parent
    if root == workspace or workspace in root.parents:
        raise ValueError("Expert data must live outside the synced repository; use " + DEFAULT_DATA_DIR)
    return root / f"mt50_v{DATA_VERSION}_s{cfg.seed}_t{cfg.demo_tail}_m{cfg.demo_max_steps}" / f"{name}.npz"


def load_expert_task(cfg, name):
    """Read a prefix of the canonical MT50 episodes; never generate during training."""
    path = dataset_path(cfg, name)
    hint = (f"Run on the pod: python generate_expert_data.py --data-dir {cfg.data_dir} "
            f"--seed {cfg.seed} --n-demos {max(250, cfg.n_demos)} "
            f"--demo-tail {cfg.demo_tail} --demo-max-steps {cfg.demo_max_steps}")
    if not path.exists():
        raise FileNotFoundError(f"Missing shared expert data: {path}. {hint}")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    lengths = data.get("lengths", np.array([]))
    if (data.get("version") != DATA_VERSION or data.get("benchmark") != "MT50"
            or data.get("task") != name or data.get("seed") != cfg.seed
            or data.get("demo_tail") != cfg.demo_tail
            or data.get("demo_max_steps") != cfg.demo_max_steps
            or lengths.ndim != 1 or lengths.dtype.kind not in "iu"
            or len(lengths) == 0 or np.any(lengths <= 0)
            or np.any(lengths > cfg.demo_max_steps)
            or "obs" not in data or "act" not in data
            or data["obs"].ndim != 2 or data["act"].ndim != 2
            or lengths.sum() != len(data["obs"]) or lengths.sum() != len(data["act"])
            or data.get("success", np.array([])).shape != lengths.shape):
        raise ValueError(f"Invalid shared trajectory archive: {path}")
    if len(lengths) < cfg.n_demos:
        raise ValueError(f"{path} has {len(lengths)} episodes, need {cfg.n_demos}. {hint}")
    lengths = lengths[:cfg.n_demos]
    end = int(lengths.sum())
    print(f"  [shared] {name:25s} {len(lengths)} trajectories ({end} transitions)")
    return {"obs": data["obs"][:end], "act": data["act"][:end],
            "lengths": lengths, "success": data["success"][:cfg.n_demos]}


def generate_or_load_demos(envs, cfg) -> dict[str, dict]:
    """Compatibility entry point: load flat views of the shared trajectories."""
    return {env.name: {k: v for k, v in load_expert_task(cfg, env.name).items()
                       if k in ("obs", "act")} for env in envs}

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
    """Load the same shared data with episode boundaries for segment training."""
    trajectories = {}
    for env in envs:
        data = load_expert_task(cfg, env.name)
        trajectories[env.name] = _split_by_lengths(data["obs"], data["act"], data["lengths"])
    return trajectories

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
