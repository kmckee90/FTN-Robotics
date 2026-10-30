"""Regression coverage for MT50 reports and evaluation isolation."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from mt50_continual_learning import train_mt50_bc as mt50
import bc as mt10


class MT50Tests(unittest.TestCase):
    def test_num_tasks_cli_limits_saved_curriculum(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            with patch('sys.argv', ['train.py', '--artifact-root', tmp, '--num-tasks', '3']), \
                 patch.object(mt50, 'train') as train:
                mt50.main()
            cfg = train.call_args.args[0]
            self.assertEqual(cfg.tasks, mt50.DEFAULT_TASKS[:3])
            saved = json.loads((Path(tmp) / 'mt50-continual/config.json').read_text())
            self.assertEqual(saved['tasks'], list(mt50.DEFAULT_TASKS[:3]))

    def test_num_tasks_custom_list_and_validation(self):
        cfg = mt50.ExpertBCConfig(tasks=('push-v3', 'reach-v3'), num_tasks=1)
        mt50.validate_config(cfg)
        self.assertEqual(cfg.tasks, ('push-v3',))
        cfg = mt50.ExpertBCConfig(num_tasks=100)
        mt50.validate_config(cfg)
        self.assertEqual(cfg.tasks, mt50.DEFAULT_TASKS)
        for count in (0, -1):
            with self.assertRaisesRegex(ValueError, 'num_tasks must be positive'):
                mt50.validate_config(mt50.ExpertBCConfig(num_tasks=count))

    def test_prior_row_precedes_training(self):
        for module in (mt50,):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = module.ExpertBCConfig(
                    tasks=('reach-v3', 'push-v3'), output_path=str(Path(tmp) / 'matrix.npz'),
                    kt_dir='', ckpt_path='', no_mask=True, bc_epochs=1, bc_batch_size=4,
                    ftn_hidden_size=16, ftn_k=4, ftn_square_dim=4,
                    ftn_layers=1, ftn_inner_size=2, lateral_steps=0,
                )
                envs = [SimpleNamespace(name=name, obs_dim=3, act_dim=2,
                                        env=unittest.mock.Mock(), rng=np.random.default_rng(0))
                        for name in cfg.tasks]
                demos = {name: {'obs': np.zeros((4, 3), dtype=np.float32),
                                'act': np.zeros((4, 2), dtype=np.float32)} for name in cfg.tasks}

                def prior(env, agent, config, demo, index):
                    self.assertFalse(agent.opt.state, 'Optimizer stepped before prior evaluation')
                    return .25

                with patch.object(module, 'build_envs', return_value=envs), \
                     patch.object(module, 'generate_or_load_demos', return_value=demos), \
                     patch.object(module, 'eval_prior_task', side_effect=prior) as evaluate_prior, \
                     patch.object(module, 'eval_task', return_value=.75):
                    matrix = module.train(cfg)
                self.assertEqual(evaluate_prior.call_count, 2)
                self.assertEqual(matrix.shape, (3, 2))
                np.testing.assert_array_equal(matrix[0], [.25, .25])
                np.testing.assert_array_equal(matrix[1:], np.full((2, 2), .75))
                with np.load(cfg.output_path) as saved:
                    self.assertEqual(list(saved['row_labels']), ['prior', *cfg.tasks])
                    self.assertEqual(int(saved['phases_done']), 2)

    def test_independent_training_and_eval_search_budgets(self):
        cfg = mt50.ExpertBCConfig(
            ftn_hidden_size=16, ftn_k=4, ftn_square_dim=4,
            ftn_layers=1, ftn_inner_size=2, lateral_steps=0,
            reconfig_steps=5, eval_reconfig_steps=2,
            eval_episodes=3, eval_max_steps=1,
        )
        agent = mt50.BCAgent(3, 2, cfg)
        agent.set_normalizer(torch.zeros(3), torch.ones(3))
        obs, act = torch.zeros(4, 3), torch.zeros(4, 2)
        demo = {'obs': np.zeros((4, 3), dtype=np.float32),
                'act': np.zeros((4, 2), dtype=np.float32)}
        env = unittest.mock.Mock()
        env.reset.return_value = (np.zeros(3, dtype=np.float32), {})
        env.step.return_value = (np.zeros(3, dtype=np.float32), 0., True, False, {'success': 1.})
        env.rng = np.random.default_rng(0)
        with patch.object(mt10, 'anneal_ste_mask', wraps=mt10.anneal_ste_mask) as search:
            agent.reconfig_actor_mask(obs, act, commit=True)
            mt10.eval_task(env, agent, cfg, reconfig_demo=demo, reconfig_every=1)
            agent.reconfig_actor_mask(obs, act, commit=True)
            self.assertEqual([call.args[3] for call in search.call_args_list], [5, 2, 2, 2, 5])
            search.reset_mock()
            mt50.evaluate_cell(env, agent, cfg, demo, 0, 1)
            self.assertEqual(search.call_args.args[3], 2)
            weights = [p.detach().clone() for p in agent.actor.parameters()]
            rng_state = torch.get_rng_state().clone()
            logits = agent.m_a_logits
            mt10.eval_prior_task(env, agent, cfg, demo, 0)
            self.assertEqual(search.call_args.args[3], 2)
            self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
            self.assertIs(agent.m_a_logits, logits)
            for old, param in zip(weights, agent.actor.parameters()):
                self.assertTrue(torch.equal(old, param))
            cfg.eval_reconfig_steps = None
            mt10.eval_task(env, agent, cfg, reconfig_demo=demo)
            self.assertEqual(search.call_args.args[3], 5)
        self.assertEqual(cfg.reconfig_steps, 5)

    def test_reconfig_budget_cli(self):
        for module in (mt50,):
            with patch('sys.argv', ['train.py', '--name', 'budget_test',
                                   '--reconfig-steps', '7', '--eval-reconfig-steps', '3']), \
                 patch.object(module, 'train') as train:
                module.main()
            cfg = train.call_args.args[0]
            self.assertEqual(cfg.reconfig_steps, 7)
            self.assertEqual(cfg.eval_reconfig_steps, 3)

    def test_no_mask_cli_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('sys.argv', ['train.py', '--artifact-root', tmp, '--no-mask']), \
                 patch.object(mt50, 'train') as train:
                mt50.main()
            self.assertTrue(train.call_args.args[0].no_mask)

    def test_named_run_output_paths(self):
        with patch('sys.argv', ['train_mt50_bc.py', '--name', 'trial_a']), \
             patch.object(mt50, 'train') as train:
            mt50.main()
        cfg = train.call_args.args[0]
        self.assertEqual(Path(cfg.output_path), Path(mt50.ExpertBCConfig().artifact_root) / 'trial_a/results.npz')
        self.assertEqual(Path(cfg.ckpt_path), Path(mt50.ExpertBCConfig().artifact_root) / 'trial_a/model.pt')
        self.assertEqual(Path(cfg.kt_dir), Path(mt50.ExpertBCConfig().artifact_root) / 'trial_a/checkpoints')

    def test_named_run_explicit_output_overrides(self):
        with patch('sys.argv', ['train_mt50_bc.py', '--name', 'trial_b',
                               '--output-path', '/tmp/custom.npz', '--kt-dir', '']), \
             patch.object(mt50, 'train') as train:
            mt50.main()
        cfg = train.call_args.args[0]
        self.assertEqual(cfg.output_path, '/tmp/custom.npz')
        self.assertEqual(cfg.ckpt_path, str(Path(mt50.ExpertBCConfig().artifact_root) / 'trial_b/model.pt'))
        self.assertEqual(cfg.kt_dir, '')

    def test_full_curriculum_and_incremental_reports(self):
        self.assertEqual(len(mt50.DEFAULT_TASKS), 50)
        self.assertEqual(set(mt50.DEFAULT_TASKS), set(mt50.metaworld.ALL_V3_ENVIRONMENTS))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = mt50.ExpertBCConfig(output_path=str(Path(tmp) / 'results.npz'))
            matrix = np.full((51, 50), np.nan, dtype=np.float32)
            matrix[0] = np.linspace(0, 1, 50)
            mt50.save_progress(matrix, cfg, 0, reports=True)
            with np.load(cfg.output_path) as data:
                self.assertEqual(data['matrix'].shape, (51, 50))
                self.assertEqual(int(data['phases_done']), 0)
                self.assertEqual(data['row_labels'][0], 'prior')
                self.assertEqual(int(data['cells_done']), 50)
                self.assertTrue(np.isnan(data['matrix'][1:]).all())
            with Path(cfg.output_path).with_suffix('.csv').open() as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows), 52)
            self.assertEqual(rows[1][0], 'prior')
            self.assertEqual(len(rows[0]), 51)
            self.assertEqual(rows[1][-1], '1.0000')
            self.assertEqual(rows[2][1:], [''] * 50)
            self.assertTrue(Path(cfg.output_path).with_suffix('.png').is_file())
            matrix[1, 0] = .7
            mt50.save_progress(matrix, cfg, 1)
            with np.load(cfg.output_path) as data:
                self.assertEqual(int(data['cells_done']), 51)
                self.assertEqual(int(data['phases_done']), 1)

    def test_eval_modes_and_state_isolation(self):
        cfg = mt50.ExpertBCConfig(ftn_hidden_size=64, ftn_k=16,
                                 ftn_square_dim=8, ftn_layers=2, ftn_inner_size=4)
        agent = mt50.BCAgent(3, 2, cfg)
        agent.stored_m_a[0] = agent.m_a.clone()
        agent.m_a_logits = torch.ones(64)
        original_logits = agent.m_a_logits
        env = type('Env', (), {'rng': np.random.default_rng(3)})()
        original_rng = env.rng
        demo = {'sentinel': True}
        weights = [p.detach().clone() for p in agent.actor.parameters()]
        for mode in ('ftn', 'stored_mask', 'fixed_mask', 'no_mask'):
            cfg.stored_mask = mode == 'stored_mask'
            cfg.fixed_mask = mode == 'fixed_mask'
            cfg.no_mask = mode == 'no_mask'
            for tj in (0, 1):
                state = torch.get_rng_state().clone()
                with patch.object(mt50, 'eval_task', return_value=.5) as evaluate:
                    self.assertEqual(mt50.evaluate_cell(env, agent, cfg, demo, 0, tj), .5)
                    kwargs = evaluate.call_args.kwargs
                    search = mode == 'ftn' or (mode == 'stored_mask' and tj == 1)
                    self.assertEqual('reconfig_demo' in kwargs, search)
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                self.assertIs(env.rng, original_rng)
                self.assertIs(agent.m_a_logits, original_logits)
                self.assertTrue(agent.actor.training)
                for old, p in zip(weights, agent.actor.parameters()):
                    self.assertTrue(torch.equal(old, p))
                    self.assertTrue(p.requires_grad)


if __name__ == '__main__':
    unittest.main()
