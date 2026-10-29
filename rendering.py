"""Shared two-panel rollout and globally consolidated FTN mask visualization."""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
import metaworld
import numpy as np
import torch
from PIL import Image, ImageDraw
from metaworld.policies import ENV_POLICY_MAP
from bc import BCAgent, disjoint_task_mask
from ftn import mask_to_grid
from runtime import DEFAULT_RUN_NAMES, configure_runtime, named_run
from spatial_permute import spatial_permutation


class MaskEMA:
    """Per-channel max-decay EMA of the active mask on the square grid.
    R decays at 0.80, G at 0.20, B at 0.875 -> active=white, recent=purple,
    old=blue (matches run_experiments.py)."""

    def __init__(self, hidden_size: int, square_dim: int, permutation=None):
        self.square_dim = square_dim
        n_blocks = hidden_size // (square_dim * square_dim)
        self.gh, self.gw = square_dim, n_blocks * square_dim
        self.ema = torch.zeros(self.gh, self.gw, 3)
        # Display-position -> original raster-position; never change policy masks.
        self.permutation = None
        if permutation is not None:
            permutation = np.asarray(permutation)
            if permutation.shape != (hidden_size,) or not np.array_equal(
                np.sort(permutation), np.arange(hidden_size)
            ):
                raise ValueError("permutation must contain every mask position once")
            self.permutation = permutation.astype(np.int64)

    def update(self, mask: torch.Tensor):
        m2d = mask_to_grid(mask.detach().float(), self.square_dim)
        self.ema[:, :, 0] = torch.max(0.80 * self.ema[:, :, 0], m2d)
        self.ema[:, :, 1] = torch.max(0.20 * self.ema[:, :, 1], m2d)
        self.ema[:, :, 2] = torch.max(0.875 * self.ema[:, :, 2], m2d)

    def image(self) -> np.ndarray:
        raster = (self.ema.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        if self.permutation is not None:
            raster = raster.reshape(-1, 3)[self.permutation].reshape(raster.shape)
        return raster

def _upscale_nn(img: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))

def compose_frame(
    render_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    title: str,
    panel: int,
    bar_h: int = 30,
) -> Image.Image:
    """Left = arm render (panel x panel), right = mask EMA centered in a
    (panel x panel) tile, with a title bar across the top."""
    # Left panel: resize render to panel x panel.
    left = Image.fromarray(render_rgb).resize((panel, panel), Image.BILINEAR)

    # Right panel: scale the mask grid up to fill the panel width, center it.
    gh, gw = mask_rgb.shape[:2]
    scale = max(1, panel // gw)
    m_up = _upscale_nn(mask_rgb, scale)
    m_img = Image.fromarray(m_up)
    right = Image.new("RGB", (panel, panel), (12, 12, 18))
    ox = (panel - m_img.width) // 2
    oy = (panel - m_img.height) // 2
    right.paste(m_img, (max(0, ox), max(0, oy)))

    canvas = Image.new("RGB", (panel * 2, panel + bar_h), (0, 0, 0))
    canvas.paste(left, (0, bar_h))
    canvas.paste(right, (panel, bar_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title, fill=(255, 255, 255))
    draw.text(
        (panel + 6, 8),
        "FTN mask (active=white -> purple -> blue)",
        fill=(180, 180, 255),
    )
    return canvas

class RenderTaskEnv:
    """One MT10 env class with offscreen rendering. Standard reset (random goal
    variant), same obs/act handling as train_mt10_bc.StandardTaskEnv."""

    def __init__(self, name, env_cls, tasks, seed, camera, height, width):
        self.name = name
        self.env = env_cls(
            render_mode="rgb_array", camera_name=camera, height=height, width=width
        )
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

    def render(self) -> np.ndarray:
        # Gymnasium 1.3's _get_viewer only restores a context when a SINGLE
        # renderer owns multiple view modes. We instead keep multiple envs
        # alive, each with its own rgb_array viewer. Restore this env's context
        # explicitly: otherwise revisiting it can use another model's GPU
        # resources and display corrupted meshes despite unchanged simulation.
        renderer = self.env.mujoco_renderer
        renderer._get_viewer(self.env.render_mode).make_context_current()
        # EGL offscreen framebuffers come out vertically flipped; un-flip them.
        return np.flipud(self.env.render()).copy()


@dataclass
class RenderConfig:
    device: str = "auto"
    num_threads: int = 3
    seed: int | None = None  # checkpoint seed unless overridden
    max_steps: int = 200
    success_tail: int = 32
    frame_stride: int = 2
    render_size: int = 320
    duration: int = 60
    camera: str = "corner3"
    control: str = "both"  # expert, ftn, or both; each output has two panels
    segment_length: int | None = None  # checkpoint value; MT50 fallback: 32
    reconfig_stride: int | None = None  # checkpoint stride; MT50 fallback: window length
    reconfig_steps: int | None = None  # checkpoint search budget
    spatial_roundness: float = 2.0
    spatial_center: float = 0.4
    spatial_radius: int = 3
    spatial_sweeps: int = 8
    spatial_temperature: float = 0.05


def resolve_checkpoint(run, experiment):
    manifest = run / "run.json"
    if manifest.exists():
        entry = json.loads(manifest.read_text())
        if entry["experiment"] != experiment:
            raise ValueError(f"{run} belongs to {entry['experiment']}, not {experiment}")
        path = (run / entry["checkpoint"]).resolve()
    else:
        path = run / "model.pt"  # earlier clean-repo checkpoints
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}. Train this --name first.")
    return path


def load_agent(path, options):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**checkpoint["cfg"])
    cfg.device, cfg.num_threads = options.device, options.num_threads
    configure_runtime(cfg)
    device = torch.get_default_device()
    agent = BCAgent(checkpoint["obs_dim"], checkpoint["act_dim"], cfg)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.actor.eval().requires_grad_(False)
    agent.set_normalizer(checkpoint["obs_mean"].to(device), checkpoint["obs_std"].to(device))
    agent.m_a = checkpoint["m_a"].to(device)
    agent.stored_m_a = {int(i): m.to(device) for i, m in checkpoint["stored_m_a"].items()}
    return agent, cfg, checkpoint


def raster_masks(masks, square_dim):
    """Convert vector-indexed mask rows to the parent's block-grid indexing."""
    h = masks.shape[1]
    order = np.arange(h).reshape(-1, square_dim, square_dim).transpose(1, 0, 2).ravel()
    return masks[:, order]


def archive_masks(checkpoint, learned_tasks):
    archive = checkpoint.get("mask_archive")
    if archive is None:
        # Older MT50 checkpoints already include their complete task masks.
        stored = checkpoint.get("stored_m_a", {})
        if stored:
            ids = sorted(stored)
            return (np.stack([stored[i].cpu().numpy() for i in ids]),
                    np.asarray(ids), "committed mask for each completed training task")
        return None
    masks = np.unpackbits(archive["packed"].cpu().numpy(), axis=1,
                          count=int(archive["hidden_size"]))
    ids = archive["task_indices"].cpu().numpy()
    if not len(masks) or len(ids) != len(masks) or np.any(ids < 0) or np.any(ids >= len(learned_tasks)):
        raise ValueError("Checkpoint mask archive does not match its learned tasks")
    return masks, ids, archive["scope"]


def reset_episode(env, task, seed):
    env.env.set_task(task)
    obs, _ = env.env.reset(seed=seed)
    return obs.astype(np.float32)


def observe_expert(env, task, task_index, agent, options, seed):
    """Recover the trailing-window mask schedule without updating actor weights."""
    cfg = agent.cfg
    length = options.segment_length or getattr(cfg, "segment_length", 32)
    stride = options.reconfig_stride or getattr(cfg, "segment_stride", length)
    obs = reset_episode(env, task, seed)
    policy = ENV_POLICY_MAP[env.name]()
    observations, actions, masks, changes = [], [], [], []
    success, first_success = False, None
    mask = None
    for step in range(options.max_steps):
        # Some scripted experts mutate observation slices: preserve the raw input.
        action = np.clip(policy.get_action(obs.copy()), -1, 1).astype(np.float32)
        observations.append(obs.copy())
        actions.append(action.copy())
        if step % stride == 0:
            if cfg.no_mask:
                mask = torch.ones(agent.actor.H)
            elif cfg.fixed_mask:
                mask = disjoint_task_mask(task_index, cfg.ftn_hidden_size, cfg.ftn_k)
            elif getattr(cfg, "stored_mask", False):
                mask = agent.stored_m_a[task_index]
            else:
                agent.m_a_logits = None
                support_obs = agent._norm(torch.tensor(np.asarray(observations[-length:]), dtype=torch.float32))
                support_act = torch.tensor(np.asarray(actions[-length:]), dtype=torch.float32)
                mask, _ = agent.reconfig_actor_mask(support_obs, support_act, commit=False,
                                                  steps=options.reconfig_steps)
            changes.append(step)
        masks.append(mask.detach().cpu().numpy().copy())
        obs, _, term, trunc, info = env.step(action)
        if info.get("success", 0) > .5:
            success = True
            if first_success is None:
                first_success = step
        if term or trunc or (first_success is not None and step - first_success >= options.success_tail):
            break
    return dict(obs=np.asarray(observations), actions=np.asarray(actions),
                masks=np.asarray(masks), changes=np.asarray(changes), success=success)


def render_episode(env, task, agent, rollout, permutation, options, seed, control, path):
    obs = reset_episode(env, task, seed)
    np.testing.assert_allclose(obs, rollout["obs"][0], atol=1e-6)
    ema = MaskEMA(agent.actor.H, agent.cfg.ftn_square_dim, permutation)
    frames, success = [], False
    for step, raw_mask in enumerate(rollout["masks"]):
        mask = torch.tensor(raw_mask, dtype=torch.float32)
        ema.update(mask)
        if step % options.frame_stride == 0:
            title = f"{control.upper()} | {env.name} | step {step}"
            frames.append(compose_frame(env.render(), ema.image(), title, options.render_size))
        action = rollout["actions"][step] if control == "expert" else agent.act(obs, mask)
        obs, _, term, trunc, info = env.step(action)
        success |= bool(info.get("success", 0) > .5)
        if term or trunc:
            break
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=options.duration,
                   loop=0, optimize=False)
    return {"file": path.name, "frames": len(frames), "success": success}


def render_run(run, experiment, options, selected_tasks=None):
    checkpoint_path = resolve_checkpoint(run, experiment)
    with checkpoint_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    agent, cfg, checkpoint = load_agent(checkpoint_path, options)
    learned = list(checkpoint["tasks"][:checkpoint["phases_done"]])
    if not learned:
        raise ValueError("The checkpoint has no completed training tasks to visualize")
    selected = list(selected_tasks) if selected_tasks else learned
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(learned):
        raise ValueError(f"--tasks must be a unique subset of learned tasks: {learned}")
    seed = cfg.seed if options.seed is None else options.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    archive = archive_masks(checkpoint, learned)
    out = run / "visualizations"
    out.mkdir(parents=True, exist_ok=True)
    # Always use the experiment's benchmark, including for a selected subset.
    benchmark = (metaworld.MT10 if experiment == "mt10" else metaworld.MT50)(seed=seed)
    rollouts = {}

    def environment(name):
        return RenderTaskEnv(name, benchmark.train_classes[name],
                             [t for t in benchmark.train_tasks if t.env_name == name],
                             seed + learned.index(name) + 1, options.camera,
                             options.render_size, options.render_size)

    def collect(name, env):
        index = learned.index(name)
        episode_seed = seed + index + 1
        task = env.tasks[int(np.random.default_rng(episode_seed).integers(len(env.tasks)))]
        # Task-specific RNGs make task subsets independent of rendering order.
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        return task, episode_seed, observe_expert(env, task, index, agent, options, episode_seed)

    if archive is None:
        # Legacy MT10 files predate segment-mask exports. Explicit fallback:
        # recover from every learned task, even if only one task is rendered.
        print("Legacy checkpoint: recovering global masks across every learned task", flush=True)
        masks, ids = [], []
        for index, name in enumerate(learned):
            env = environment(name)
            try:
                record = collect(name, env)
                rollouts[name] = record
                recovered = record[2]["masks"][record[2]["changes"]]
                masks.extend(recovered)
                ids.extend([index] * len(recovered))
            finally:
                env.env.close()
        archive = np.asarray(masks), np.asarray(ids), "legacy fallback: recovered windows from every learned task"
    masks, ids, scope = archive
    grids = raster_masks(masks, cfg.ftn_square_dim)
    spatial = dict(roundness=options.spatial_roundness, center=options.spatial_center,
                   radius=options.spatial_radius, sweeps=options.spatial_sweeps,
                   temperature=options.spatial_temperature)
    print(f"Fitting one global blob layout: {len(masks)} masks across {len(set(ids.tolist()))} tasks", flush=True)
    permutation = spatial_permutation(grids, (cfg.ftn_square_dim, agent.actor.H // cfg.ftn_square_dim), **spatial)
    metadata = dict(checkpoint=os.path.relpath(checkpoint_path, out), sha256=digest,
                    fit_tasks=learned, scope=scope, mask_count=len(masks), options=spatial,
                    indexing="display position -> original mask_to_grid raster position")
    np.savez_compressed(out / "layout.npz", permutation=permutation,
                        packed_masks=np.packbits(grids > 0, axis=1), hidden_size=agent.actor.H,
                        task_indices=ids, metadata=json.dumps(metadata))
    report = dict(name=run.name, experiment=experiment, checkpoint=metadata["checkpoint"],
                  sha256=digest, layout="layout.npz", global_scope=scope,
                  options=asdict(options), results=[])
    controls = ("expert", "ftn") if options.control == "both" else (options.control,)
    for name in selected:
        env = environment(name)
        try:
            task, episode_seed, rollout = rollouts[name] if name in rollouts else collect(name, env)
            np.savez_compressed(out / f"{name}_masks.npz", **rollout)
            result = {"task": name, "expert_success": rollout["success"]}
            for control in controls:
                path = out / f"{name}_{control}.gif"
                result[control] = render_episode(env, task, agent, rollout, permutation,
                                                options, episode_seed, control, path)
                print(f"Saved {path}", flush=True)
            report["results"].append(result)
            (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        finally:
            env.env.close()
    return report


def visualization_cli(experiment, script_path, defaults=None):
    defaults = defaults or RenderConfig()
    parser = argparse.ArgumentParser(description="Demo left / FTN mask right, one global blob layout.")
    parser.add_argument("--name", default=DEFAULT_RUN_NAMES[experiment])
    parser.add_argument("--artifact-root", default=str(Path(script_path).resolve().parent / "artifacts"))
    parser.add_argument("--tasks", nargs="+", help="Render a subset; global layout still uses all learned masks.")
    for field, value in asdict(defaults).items():
        parser.add_argument("--" + field.replace("_", "-"), default=value,
                            type=type(value) if value is not None else int)
    args = vars(parser.parse_args())
    name, root, tasks = args.pop("name"), args.pop("artifact_root"), args.pop("tasks")
    options = RenderConfig(**args)
    for field in ("num_threads", "max_steps", "frame_stride", "render_size", "duration", "spatial_radius"):
        if getattr(options, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    for field in ("segment_length", "reconfig_stride"):
        if getattr(options, field) is not None and getattr(options, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if options.success_tail < 0 or options.spatial_sweeps < 0 or (options.reconfig_steps is not None and options.reconfig_steps < 0):
        parser.error("tail, sweeps, and reconfiguration steps must be nonnegative")
    if options.control not in ("expert", "ftn", "both"):
        parser.error("--control must be expert, ftn, or both")
    try:
        run = named_run(root, name)
        resolve_checkpoint(run, experiment)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    render_run(run, experiment, options, tasks)
