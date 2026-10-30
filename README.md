# FTN behavior cloning on MetaWorld

Two independent experiments sharing a small FTN and behavior-cloning library:

- `mt50_continual_learning/train_mt50_bc.py`: sequential training across all 50 MetaWorld tasks, with an untrained prior and evaluation of every task after every training phase.
- `mt10_segment_representation/train_mt10_transitions_bc_par.py`: training on contiguous windows from all 10 MetaWorld tasks, jointly searching a mask per segment and updating shared actor weights in parallel batches.

Each script defines its experiment configuration near the top. Portable model presets live in `models/presets/*.yaml`; `models/config.py` defines their schema and fallback defaults. Neither imports another experiment. No planning code or old checkpoints are required.

## Setup

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Run setup inside the pod, from the synced repository root:

```sh
cd /home/dev/workspace
export UV_PROJECT_ENVIRONMENT=/home/dev/venvs/ftn-bc
bash setup_pod.sh
uv run python -u generate_expert_data.py
uv run python mt50_continual_learning/train_mt50_bc.py --smoke
uv run python mt10_segment_representation/train_mt10_transitions_bc_par.py --smoke
```

MetaWorld, MuJoCo, environments, and robot/task assets are installed through the dependencies. Generate the shared dataset **on the pod** once, from the repository root:

```sh
uv run python -u generate_expert_data.py
```

Generation uses three independent CPU worker processes by default, matching this pod’s reserved CPUs. Set `--workers 1` for sequential generation or `--workers 6` to use more CPU capacity (the current pod limit is 12 CPUs). Workers generate separate task groups, reuse their own MetaWorld benchmark, and limit native math-library threads to one. Task seeds and episode ordering are independent of worker count, and existing archives remain reusable.

This generates 250 scripted-expert episodes for each of the 50 MT50 tasks, with a 400-step horizon and 100-step success tail. Data lives at `/home/dev/ftn_data/mt50_experts`, outside `/home/dev/workspace`, so it stays on the pod home volume and never syncs locally. `--data-dir` can select another pod path outside the repository; pass the same path to both training scripts. Repository-local dataset paths are rejected. `.actlignore` also excludes legacy `data_cache/` directories; restart an existing sync session to activate that exclusion. Deleting the old local `data_cache` does not affect the new dataset.

Each task archive stores observations, actions, episode lengths, success flags, and generation metadata. MT10 segments needs the lengths to keep windows within individual episodes; MT50 reads a flattened view of the same arrays. MT10 uses the first 150 episodes by default and MT50 all 250; set `--n-demos 250` on MT10 to use exactly the same episodes. The shared demonstrations use MT50 goal variants for both experiments (MT10 still constructs its MT10 environments).

Training only reads data and fails with a generation command if it is missing or too small. Smoke runs use smaller prefixes of this same dataset. Seed, horizon, and success tail select a dataset version; changing these requires generating the corresponding version first. `--tasks reach-v3` generates just that task for a quick check. Re-running the generator skips completed tasks; increasing `--n-demos` regenerates smaller task files deterministically, preserving their episode prefix. Failed expert attempts are retained, as before, and success counts are printed. Files are locked during generation and published atomically so interrupted tasks can be rerun safely.

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
uv run python mt50_continual_learning/train_mt50_bc.py --name first10 --num-tasks 10
uv run python mt10_segment_representation/train_mt10_transitions_bc_par.py --name random --batch-mode random --segment-batch 64
```

Defaults preserve the original experiments: MT50 begins with the original eight-task curriculum, then appends the remaining 42 tasks alphabetically; MT10 uses all ten tasks and contiguous segment batches. MT50 supports searched, stored, fixed, and unmasked baselines. The default hidden size cannot fit 50 disjoint masks of 256 units: fixed-mask runs need a smaller `ftn_k` or a larger compatible hidden size.

MT50 `--num-tasks N` selects `tasks[:N]` before data loading, training, evaluation, and saving the resolved configuration. It also applies to an explicit `--tasks` list or the smoke curriculum. Omit it to use all configured tasks; N must be positive, and values larger than the list keep the whole list.

MT50 evaluation searches masks using expert support, including for future tasks; its transfer scores are **not zero-shot**. Both experiments fit a frozen observation normalizer using demonstrations from all configured tasks. MT10 produces training losses and a trained actor; it does not perform a success-rate evaluation.

## Model alternatives

Choose a model preset with `--model-config FILENAME` in either experiment:

```sh
uv run python mt50_continual_learning/train_mt50_bc.py --model-config bayeslora.yaml --num-tasks 10 --name bayeslora-first10
uv run python mt10_segment_representation/train_mt10_transitions_bc_par.py --model-config ftn_reactive_low_noreplay.yaml --name ftn-reactive
```

Bare filenames are resolved under `models/presets/`, independently of your working directory. The `.yaml` extension is optional: `--model-config bayeslora` also works. Explicit paths remain supported (for example `--model-config ./custom.yaml` or an absolute path).

The six self-contained presets are:

| YAML under `models/presets/` | Model | Inference / replay |
|---|---|---|
| `ftn.yaml` | FTN | Scheduled search, MT50 support replay |
| `lora.yaml` | LoRA | Scheduled search, MT50 support replay |
| `bayesftn.yaml` | BayesFTN | Reactive posterior updates, MT50 support replay |
| `bayeslora.yaml` | BayesLoRA | Reactive posterior updates, MT50 support replay |
| `ftn_reactive_low_noreplay.yaml` | FTN | Loss-triggered search, SD 5 / EMA 0.9, no replay |
| `lora_reactive_low_noreplay.yaml` | LoRA | Loss-triggered search, SD 5 / EMA 0.9, no replay |

Copy any preset and edit its plain YAML mapping. Keys use Python field names (`lora_rank`, `bayes_temperature`, etc.); each file must specify `model`. Presets configure architecture, mask search, and replay, while tasks, data, training duration/batching, optimizer learning rate, and output paths remain experiment options. Unknown keys, duplicate keys, and incorrect value types are rejected. The supplied presets explicitly use a common bank size of 2304, k=256, and a 48×48 layout in both experiments; change those three values together for a smaller bank.

Precedence is **fallback/experiment defaults → YAML → smoke overrides → explicit CLI flags**. For example, `--model-config lora.yaml --lora-rank 8` overrides the YAML rank, and adding `--smoke` shrinks the bank/search budgets without changing the selected model. Fields not changed by the smoke configuration keep their YAML values. MT10 then disables archive-based support replay (`consolidate: false`, `replay_max_chunks: 0`) regardless of preset/CLI, because it uses segment co-training instead. Reusing segment batches across cycles/epochs remains enabled; that is not archive replay. Reactive presets keep stable masks until the detector fires (LoRA continuous reactive updates are off).

Each run saves the selected source file as `model_config.yaml`, records it in `run.json`, and saves all effective values—including overrides—in `config.json` and the checkpoint. Reloading a checkpoint does not depend on the YAML remaining at its original path. Omitting `--model-config` preserves the previous experiment defaults and direct `--model` selection.

Both experiments accept `--model ftn` (the unchanged default), `--model lora`, `--model bayesftn`, or `--model bayeslora`. Names are case-insensitive; `bayes`, `bayes_ftn`, and `bayes_lora` are also accepted. All use the same shared demonstrations, action bounds, normalization, checkpoint format, and mask visualizations. MT50 baselines (`--no-mask`, `--fixed-mask`, `--stored-mask`) work with either backbone; bypassing mask search also bypasses Bayesian updates.

```sh
uv run python mt50_continual_learning/train_mt50_bc.py --model bayeslora --num-tasks 10 --name bayeslora-first10
uv run python mt10_segment_representation/train_mt10_transitions_bc_par.py --model lora --name lora-segments
uv run python mt10_segment_representation/train_mt10_transitions_bc_par.py --model bayesftn --bayes-changepoint-reconfig-steps 40 --bayes-changepoint-prior-decay 0 --name bayes-segments
```

`LoRA` changes the backbone: a frozen random MLP has a trainable rank-r atom bank on **every** linear layer, including the output. One k-hot mask gates the same atoms throughout the network. Atom deltas and optional atom biases are scaled by `alpha/k`; nonzero B/bias initialization makes initially inactive atoms reachable by mask search. Every atom receives a mask-gradient path. `--lora-layers`, `--lora-width`, `--lora-rank`, `--lora-alpha`, `--lora-weight-scale`, `--lora-b0-scale`, `--lora-dropout`, and `--lora-bias`/`--no-lora-bias` control these features. Defaults use the upstream constructor sizes (4 layers, width 64, rank 4, alpha 16), rather than the much larger upstream benchmark presets.

LoRA search supports `--lora-reconfig-algorithm annealste|fast`, `--lora-reconfig-optimizer Adam|SGD|RMSprop`, `--reconfig-noise-scale`, warm logits (`--lora-warm-start-reconfig` or the experiment's `--mask-warm-start`), and the existing lateral smoothing parameters. With annealed search, `--lora-match-stored` enables nearest-mask recall, with `--lora-match-stored-after` and `--lora-match-stored-topk` controlling timing and breadth. MT50 recalls stored task masks; MT10 recalls previously inferred segment masks. Cold evaluation excludes stored-mask matching and overlap penalties. The optional `--reconfig-overlap-mode bump|gated|seen_logits`, `--reconfig-overlap-lambda`, `--reconfig-overlap-threshold`, and `--reconfig-overlap-temperature` expose both upstream overlap cost families. `seen_logits` uses the union of atoms used during weight updates. Classification-only output entropy and classification heads are not applied to continuous-action MSE regression.

`BayesFTN` keeps the FTN backbone; `BayesLoRA` uses the LoRA backbone. Both replace independent mask search with a persistent approximate posterior. Logits are inference buffers, excluded from the weight optimizer. Updates use exact-k Gumbel masks, a sigmoid straight-through gradient, and factorized Bernoulli KL to the detached, optionally decayed prior. Configure ordinary updates with:

- `--bayes-reconfig-steps`, `--bayes-reconfig-lr`, `--bayes-reconfig-sample-size` (default: all supplied support).
- `--bayes-kl-weight`, `--bayes-temperature`, `--bayes-prior-decay`.
- `--bayes-stochastic-mask`/`--no-bayes-stochastic-mask`, `--bayes-reconfig-take-best`, and `--bayes-sample-type gumbel|annealste`.

Bayesian models default to reactive inference: the first support batch or a loss spike triggers a reset/refit, while stable batches receive cheap posterior updates. The detector compares pre-update loss to an EMA mean plus `--reconfig-reactive-sd` standard deviations, with EMA factor `--reconfig-reactive-ema` and a `ceil(1/(1-EMA))` warmup after each trigger. It observes support loss, not task labels or task boundaries. Post-reconfiguration loss updates its statistics. Set `--no-reconfig-reactive` to use MT50's `--reconfig-train` schedule (or MT10's segment cycles) instead. LoRA can opt into the detector with `--reconfig-reactive`, `--lora-reactive-continuous-updates`, and `--lora-changepoint-reconfig-steps`.

Changepoint posterior settings are independent of the ordinary-update settings:

- `--bayes-changepoint-reconfig-steps` (30), `--bayes-changepoint-reconfig-lr` (0.3), `--bayes-changepoint-reconfig-sample-size` (all supplied support).
- `--bayes-changepoint-kl-weight` (0), `--bayes-changepoint-temperature` (1), `--bayes-changepoint-prior-decay` (0).
- `--bayes-changepoint-stochastic-mask` / `--no-bayes-changepoint-stochastic-mask` (enabled).

MT50 carries one posterior/detector through its task stream. MT10 keeps independent posterior/detector rows keyed by **segment identity**, across cycles and epochs; reordered batches cannot transfer one segment's posterior to another. This preserves the experiment's independent masks. New segment rows trigger their own initial refits. The state bank resides on CPU, with only active rows moved to the training device. Its logits and masks require roughly `8 × number_of_segments × hidden_size` bytes, plus detector state, and are included in checkpoints.

Weight updates enforce strict inactive-unit isolation for all four models. `weight_decay` must be zero (nonzero values are rejected). Every committed mask search clears the weight optimizer's Adam moments and step counters; a reactive call that holds its mask without searching preserves them. Every weight step also clears gradients and Adam moments for units inactive in **all** masks contributing to that step. Thus replay may intentionally train units inactive in the current task's mask, and parallel segments may train units active in another segment, but units outside their union stay unchanged. This also protects units when replay membership or segment batches change without reconfiguration.

FTN has independent per-unit MLP parameters and per-unit readout columns, with no shared trainable output bias. LoRA has per-atom A/B/bias parameters and a frozen shared backbone; its active atoms interact through shared activations, but there are no shared trainable backbone parameters. Observation normalization is frozen. Mask inference intentionally differentiates logits for inactive candidates while actor weights remain frozen; those search gradients do not update inactive weights.

LoRA and Bayesian evaluation use stateless zero-logit searches with no KL or training state. `--eval-reconfig-lr`, `--eval-reconfig-sample-size`, `--eval-reconfig-temperature`, `--eval-reconfig-restarts`, and `--eval-reconfig-take-best` control them; MT50 also keeps its existing `--eval-reconfig-steps` budget. Each restart considers its final deterministic top-k mask, plus sampled candidates when take-best is enabled. Legacy FTN checkpoints still load; new checkpoints include architecture settings, posterior/detector state, warm logits, and learned masks. These remain model exports, not optimizer-complete training resumes.

Model code lives in `models/`: `ftn.py` (original backbone/search), `lora.py` (atom banks), `bayes.py` (posterior, cold search, detector, state banks), `search.py` (training/inference integration), and `config.py` (shared options/validation). The implementation is adapted from [the upstream LoRA model](https://github.com/kmckee90/functional_task_networks/blob/678ea643392cdde2517b916db465cc05747c32ac/ftn/lora_ftn.py) and [the Bayesian project](https://github.com/kmckee90/functional_task_networks/tree/678ea643392cdde2517b916db465cc05747c32ac/ftn/projects/bayesian_adv), pinned to commit `678ea643392cdde2517b916db465cc05747c32ac`. The robotics adapters add tanh actions, vectorized independent segment searches, frozen-backbone protection, and side-effect-free evaluation. No external source checkout is required at runtime.

Run the unit tests on the pod with `python -m unittest discover -s tests -v`. `python tests/smoke_models.py --output /home/dev/ftn_model_validation` explicitly exercises all four choices in both experiments using real shared demonstrations and tiny budgets; it checks execution, not learning quality.

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

Regular runs default to `artifacts/mt10-segments/` or `artifacts/mt50-continual/`. Smoke runs append `-smoke` to that experiment name. Use a distinct `--name` to preserve each run. Reusing a name overwrites its outputs; checkpoints are model exports, not optimizer/RNG-complete resume files. Set MT50 `--kt-dir ''` to disable per-phase snapshots. Cached demonstrations and both artifact trees are ignored by Git. Shared dataset keys include schema version, seed, horizon, and success tail; each task file preserves episode boundaries and supports smaller episode prefixes.

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

Shared model package: `models/`. Shared root modules: `bc.py` (cloning and evaluation), `data.py` (environments and demonstrations), `visualization.py` (plots), `runtime.py` (CLI and validation), `rendering.py` (two-panel videos), and `spatial_permute.py` (global blob layout). `pyproject.toml` and `uv.lock` describe the standalone environment.
