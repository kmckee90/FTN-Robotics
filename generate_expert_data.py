"""Generate the canonical MT50 trajectory dataset on the pod, outside source sync."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import fcntl
import os
import random
import tempfile

import metaworld
import numpy as np
from metaworld.policies import ENV_POLICY_MAP

from data import (DEFAULT_DATA_DIR, DATA_VERSION, StandardTaskEnv,
                  _rollout_expert, dataset_path, load_expert_task)


@contextmanager
def task_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def generate(cfg):
    names = sorted(metaworld.ALL_V3_ENVIRONMENTS)
    selected = cfg.tasks or names
    unknown = set(selected) - set(names)
    if unknown:
        raise ValueError(f"Unknown MT50 tasks: {sorted(unknown)}")
    # Validate location before constructing the benchmark or writing anything.
    dataset_path(cfg, selected[0])
    workers = getattr(cfg, 'workers', 3)
    if workers < 1:
        raise ValueError('workers must be positive')
    if len(selected) != len(set(selected)):
        raise ValueError('tasks must be unique')
    workers = min(workers, len(selected))
    print(f'Generating {len(selected)} tasks with {workers} CPU worker(s)', flush=True)
    if workers == 1:
        _generate_tasks(cfg, selected)
        return
    # Spawn clean processes: never fork an initialized MuJoCo or Torch runtime.
    # Set native-library limits before child imports to avoid nested thread pools.
    limits = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS')
    previous = {key: os.environ.get(key) for key in limits}
    try:
        for key in limits:
            os.environ[key] = '1'
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            # One benchmark per worker, reused across its tasks. Only config/names
            # cross process boundaries; trajectory arrays are written by workers.
            futures = [pool.submit(_generate_tasks, cfg, selected[i::workers])
                       for i in range(workers)]
            for future in as_completed(futures):
                future.result()  # Propagate failures: never report partial data as complete.
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _generate_tasks(cfg, selected):
    names = sorted(metaworld.ALL_V3_ENVIRONMENTS)
    benchmark = None
    for name in selected:
        path = dataset_path(cfg, name)
        with task_lock(path):
            if path.exists():
                with np.load(path, allow_pickle=False) as existing:
                    enough = len(existing['lengths']) >= cfg.n_demos
                if enough:
                    load_expert_task(cfg, name)
                    continue
            if benchmark is None:
                benchmark = metaworld.MT50(seed=cfg.seed)
            # Stable per-task streams make subsets, ordering and resume equivalent.
            task_seed = cfg.seed + names.index(name) + 1
            random.seed(task_seed)
            np.random.seed(task_seed)
            env = StandardTaskEnv(name, benchmark.train_classes[name],
                                  [t for t in benchmark.train_tasks if t.env_name == name],
                                  task_seed)
            obs, actions, success = [], [], []
            try:
                env.env.reset(seed=task_seed)
                policy = ENV_POLICY_MAP[name]()
                for episode in range(cfg.n_demos):
                    o, a, ok = _rollout_expert(env, policy, cfg)
                    obs.append(o)
                    actions.append(a)
                    success.append(ok)
                    if (episode + 1) % 25 == 0:
                        print(f'{name}: {episode + 1}/{cfg.n_demos}', flush=True)
            finally:
                env.env.close()
            # Atomic publication: readers never see a partially written task.
            fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.npz')
            try:
                with os.fdopen(fd, 'wb') as output:
                    np.savez_compressed(output, obs=np.concatenate(obs), act=np.concatenate(actions),
                                        lengths=np.array([len(o) for o in obs], dtype=np.int64),
                                        success=np.array(success, dtype=np.bool_), version=DATA_VERSION,
                                        benchmark='MT50', task=name, seed=cfg.seed,
                                        demo_tail=cfg.demo_tail, demo_max_steps=cfg.demo_max_steps,
                                        metaworld_version='3.0.0')
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            print(f'Saved {path}: {sum(success)}/{cfg.n_demos} succeeded', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--workers', type=int, default=3,
                        help='CPU processes (default: 3); use 1 for sequential generation.')
    parser.add_argument('--n-demos', type=int, default=250)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--demo-max-steps', type=int, default=400)
    parser.add_argument('--demo-tail', type=int, default=100)
    parser.add_argument('--tasks', nargs='+', help='Optional subset for testing/resuming; default: all 50.')
    cfg = parser.parse_args()
    if cfg.workers < 1 or cfg.n_demos < 1 or cfg.demo_max_steps < 1 or cfg.demo_tail < 0 or cfg.seed < 0:
        parser.error('workers, n-demos and demo-max-steps must be positive; seed and demo-tail nonnegative')
    generate(cfg)


if __name__ == '__main__':
    main()
