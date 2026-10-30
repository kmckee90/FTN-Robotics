"""Shared archive identity, episode boundaries, and resumable generation."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import data
import generate_expert_data as generator


class SharedDataTests(unittest.TestCase):
    def test_shared_views_prefix_resume_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(data_dir=tmp, seed=0, n_demos=3,
                                  demo_tail=1, demo_max_steps=10, tasks=['reach-v3'])
            class FakeEnv:
                def __init__(self, *args):
                    self.env = self
                def reset(self, **kwargs):
                    pass
                def close(self):
                    pass
            def rollout(*args):
                size = np.random.randint(2, 8)
                return np.random.rand(size, 3), np.random.rand(size, 2), 1
            benchmark = SimpleNamespace(train_classes={'reach-v3': object}, train_tasks=[])
            with patch.object(generator.metaworld, 'MT50', return_value=benchmark), \
                 patch.object(generator, 'StandardTaskEnv', FakeEnv), \
                 patch.object(generator, '_rollout_expert', side_effect=rollout) as roll:
                generator.generate(cfg)
                first = data.load_expert_task(cfg, 'reach-v3')
                generator.generate(cfg)
                self.assertEqual(roll.call_count, 3)
                cfg.n_demos = 5
                generator.generate(cfg)
                cfg.n_demos = 3
                envs = [SimpleNamespace(name='reach-v3')]
                flat = data.generate_or_load_demos(envs, cfg)['reach-v3']
                trajectories = data.generate_or_load_trajectories(envs, cfg)['reach-v3']
                np.testing.assert_array_equal(first['obs'], flat['obs'])
                np.testing.assert_array_equal(np.concatenate([o for o, _ in trajectories]), flat['obs'])
                np.testing.assert_array_equal(np.concatenate([a for _, a in trajectories]), flat['act'])
                self.assertEqual([len(o) for o, _ in trajectories], list(first['lengths']))
                cfg.n_demos = 6
                with self.assertRaisesRegex(ValueError, 'need 6'):
                    data.load_expert_task(cfg, 'reach-v3')
                cfg.seed = 1
                with self.assertRaisesRegex(FileNotFoundError, 'generate_expert_data.py'):
                    data.load_expert_task(cfg, 'reach-v3')

    def test_reject_synced_data_destination(self):
        cfg = SimpleNamespace(data_dir=str(Path(data.__file__).parent / 'data_cache'))
        with self.assertRaisesRegex(ValueError, 'outside the synced'):
            data.dataset_path(cfg, 'reach-v3')

    def test_invalid_workers_and_duplicate_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(data_dir=tmp, seed=0, demo_tail=1,
                                  demo_max_steps=10, tasks=['reach-v3'], workers=0)
            with self.assertRaisesRegex(ValueError, 'workers must be positive'):
                generator.generate(cfg)
            cfg.workers = 2
            cfg.tasks = ['reach-v3', 'reach-v3']
            with self.assertRaisesRegex(ValueError, 'tasks must be unique'):
                generator.generate(cfg)

    def test_experiment_defaults_share_dataset(self):
        from mt10_segment_representation.train_mt10_transitions_bc_par import ParSegmentBCConfig, smoke_config
        from mt50_continual_learning.train_mt50_bc import ExpertBCConfig, smoke_config as smoke50
        configs = [ParSegmentBCConfig(), ExpertBCConfig(), smoke_config(), smoke50()]
        self.assertEqual(len({str(data.dataset_path(c, 'reach-v3')) for c in configs}), 1)


if __name__ == '__main__':
    unittest.main()
