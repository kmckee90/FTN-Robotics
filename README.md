# FTN behavior cloning on MetaWorld

Two independent experiments sharing a small FTN and behavior-cloning library:

- `mt50_continual_learning/train_mt50_bc.py`: sequential training across all 50 MetaWorld tasks, with an untrained prior and evaluation of every task after every training phase.
- `mt10_segment_representation/train_mt10_transitions_bc_par.py`: training on contiguous windows from all 10 MetaWorld tasks, jointly searching a mask per segment and updating shared actor weights in parallel batches.

Each script defines its entire experiment configuration near the top, followed by its smoke configuration and training code. Neither imports another experiment. No planning code or old checkpoints are required.

## Setup

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Run setup inside the pod, from the synced repository root:

```sh
cd /home/dev/workspace
export UV_PROJECT_ENVIRONMENT=/home/dev/venvs/ftn-bc
bash setup_pod.sh
uv run python mt50_continual_learning/train_mt50_bc.py --smoke
uv run python mt10_segment_representation/train_mt10_transitions_bc_par.py --smoke
```

MetaWorld, MuJoCo, the MT10/MT50 environments and their packaged robot/task assets are installed through the dependencies. On first use, each experiment generates its own demonstrations with MetaWorld's scripted expert policies and caches them in `data_cache/`. No external dataset, parent repository, or manual model download is needed. Data generation can take a while at the full defaults.

Training uses CUDA when available and otherwise CPU. Set `--device cpu` or `--device cuda` explicitly if desired. MuJoCo simulation and expert demonstration generation run on CPU. Training and PNG plots require no display or OpenGL rendering context. On minimal Linux images, MuJoCo may need the system OpenGL runtime (`libgl1`).

### Workspace pod

Keep source local and run all setup/training in the pod's `/home/dev/workspace`. `actl` sync carries source changes. Keep the virtual environment outside the synced tree:

```sh
cd /home/dev/workspace
export UV_PROJECT_ENVIRONMENT=/home/dev/venvs/ftn-bc
bash setup_pod.sh
uv run python -m unittest discover -s tests -v
uv run python mt50_continual_learning/train_mt50_bc.py --smoke
uv run python mt10_segment_representation/train_mt10_transitions_bc_par.py --smoke
```

## Full experiments

```sh
uv run python -u mt50_continual_learning/train_mt50_bc.py --name mt50-seed0
uv run python -u mt10_segment_representation/train_mt10_transitions_bc_par.py --name mt10-seed0
```

Edit the dataclass at the top of either script, or override any field via `--field-name value`. `--help` lists all fields. Boolean fields support enabling/disabling flags, for example `--consolidate` / `--no-consolidate`. `--smoke` uses the smaller configuration defined in that same script; explicit command-line overrides apply afterward.

```sh
uv run python mt50_continual_learning/train_mt50_bc.py --name small --tasks reach-v3 push-v3 --n-demos 10 --bc-epochs 2
uv run python mt10_segment_representation/train_mt10_transitions_bc_par.py --name random --batch-mode random --segment-batch 64
```

Defaults preserve the original experiments: MT50 begins with the original eight-task curriculum, then appends the remaining 42 tasks alphabetically; MT10 uses all ten tasks and contiguous segment batches. MT50 supports searched, stored, fixed, and unmasked baselines. The default hidden size cannot fit 50 disjoint masks of 256 units: fixed-mask runs need a smaller `ftn_k` or a larger compatible hidden size.

MT50 evaluation searches masks using expert support, including for future tasks; its transfer scores are **not zero-shot**. Both experiments fit a frozen observation normalizer using demonstrations from all configured tasks. MT10 produces training losses and a trained actor; it does not perform a success-rate evaluation.

## Artifacts

Default destinations are relative to each script, regardless of your shell's working directory:

```text
mt50_continual_learning/artifacts/<name>/
  run.json             name and checkpoint location used by visualization
  config.json          resolved configuration
  results.npz          incremental (tasks + 1) × tasks success matrix
  results.config.json  configuration alongside results
  results.csv          labeled matrix
  results.txt          readable matrix
  results.png          heatmap
  model.pt             latest actor, normalizer, masks, and configuration
  checkpoints/         prior and per-phase model snapshots
  visualizations/      two-panel GIFs, mask schedules, global layout, report
mt10_segment_representation/artifacts/<name>/
  run.json             name and checkpoint location used by visualization
  config.json          resolved configuration
  model.pt             trained actor, normalizer, and configuration
  model.config.json    configuration alongside checkpoint
  results.npz          per-epoch training losses and segment counts
  loss.png             training loss curve
  visualizations/      two-panel GIFs, mask schedules, global layout, report
```

Regular runs default to `artifacts/mt10-segments/` or `artifacts/mt50-continual/`. Smoke runs append `-smoke` to that experiment name. Use a distinct `--name` to preserve each run. Reusing a name overwrites its outputs; checkpoints are model exports, not optimizer/RNG-complete resume files. Set MT50 `--kt-dir ''` to disable per-phase snapshots. Cached demonstrations and both artifact trees are ignored by Git. Cache keys include task, demonstration count, seed, horizon, and success tail; trajectory caches also preserve episode boundaries.

## Checkpoint visualizations

Use the same `--name` for training and visualization. Run these commands on the pod with the same `UV_PROJECT_ENVIRONMENT` as setup:

```sh
uv run python -u mt10_segment_representation/train_mt10_transitions_bc_par.py --name whatever0
uv run python -u mt10_segment_representation/visualize.py --name whatever0

uv run python -u mt50_continual_learning/train_mt50_bc.py --name whatever0
uv run python -u mt50_continual_learning/visualize.py --name whatever0
```

Both experiments have separate artifact roots, so the same name can be used in both. With no `--name`, each visualization script picks up its experiment's default training run (`mt10-segments` or `mt50-continual`). To render a default smoke run, pass `--name mt10-segments-smoke` or `--name mt50-continual-smoke`. `run.json` records the checkpoint location, including custom training checkpoint paths. If you override `--artifact-root`, pass the same root to visualization. Default checkpoint references are relative, so a complete run folder can be moved with the repository.

Each selected task produces two GIFs in `<experiment>/artifacts/<name>/visualizations/`:

- `<task>_expert.gif`: scripted expert demo on the left; the FTN's recovered mask on the right.
- `<task>_ftn.gif`: FTN-controlled rollout on the left, replaying the same expert-derived mask schedule on the right. This is expert-assisted mask replay, not autonomous task selection.

Both GIFs have exactly two panels, using the parent's white/purple/blue mask trail. The mask is cold-searched from a trailing observation/action window; MT10 uses its checkpoint's segment length/stride, and MT50 defaults to a 32-transition window/stride. Stored-mask, fixed-mask, and no-mask MT50 baselines retain their respective mask behavior. Actor weights stay frozen during visualization.

The spatial layout uses the parent's **global blob** objective and defaults (`roundness=2`, `center=0.4`, `radius=3`, `sweeps=8`, `temperature=0.05`). It fits one display-only permutation across **all exported learned masks**, retaining their original overlaps. MT10 exports the last committed mask for every training segment in the final epoch; MT50 exports the committed mask for every completed task. These are the masks used during learning, not newly retrained masks. The renderer uses coactivation statistics to evaluate the same global objective without a per-segment cost on every spatial swap.

`--tasks` limits the videos, **not** the global layout's fitting set. `layout.npz` records the shared permutation, packed fitting masks, task indices, and checkpoint SHA-256; `<task>_masks.npz` records the demonstration and recovered mask schedule; `results.json` records outputs and rollout outcomes. Older MT10 checkpoints without exported segment masks fall back to recovering windows across every learned task, explicitly labeled in the report.

```sh
uv run python mt10_segment_representation/visualize.py --name whatever0 --tasks drawer-open-v3 --control expert
uv run python mt50_continual_learning/visualize.py --name whatever0 --max-steps 500 --frame-stride 2
```

Visualization settings live in the `RenderConfig` at the top of each `visualize.py` entry point and the shared dataclass in `rendering.py`; `--help` lists all overrides. Rendering uses headless EGL by default on the GPU pod (`MUJOCO_GL=egl`); it requires the image's EGL/OpenGL runtime. Training itself does not render. `setup_pod.sh` installs missing EGL/OpenGL libraries on Debian/Ubuntu pod images, then installs the locked Python environment outside the synced source tree. Run it again after a fresh pod image; system packages live in the container, while the virtual environment lives in the pod home volume. GIF dependencies are included in the locked environment.

Shared root modules: `ftn.py` (actor and mask search), `bc.py` (cloning and evaluation), `data.py` (environments and demonstrations), `visualization.py` (plots), `runtime.py` (CLI and validation), `rendering.py` (two-panel videos), and `spatial_permute.py` (global blob layout). `pyproject.toml` and `uv.lock` describe the standalone environment.
