"""Model compatibility, frozen LoRA weights, persistent inference, and isolation."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from bc import BCAgent, save_checkpoint
from models.bayes import bernoulli_kl, posterior_update, detect_changepoints, update_detector, topk
from models.config import validate_model_config
from models.lora import LoRALinear
from models.search import lora_search
from rendering import load_agent, RenderConfig
from mt10_segment_representation import train_mt10_transitions_bc_par as mt10
from mt50_continual_learning import train_mt50_bc as mt50


MODELS = ('ftn', 'lora', 'bayesftn', 'bayeslora')


def config(model, segments=False, **kwargs):
    cls = mt10.ParSegmentBCConfig if segments else mt50.ExpertBCConfig
    params = dict(model=model, device='cpu', num_threads=1,
                  ftn_hidden_size=16, ftn_k=4, ftn_square_dim=4, ftn_layers=1,
                  ftn_inner_size=3, lora_layers=1, lora_width=6, lora_rank=2,
                  reconfig_steps=3, bayes_reconfig_steps=3, bayes_changepoint_reconfig_steps=3)
    params.update(kwargs)
    return cls(**params)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.get_default_device()
        torch.set_default_device('cpu')
        torch.manual_seed(9)

    def tearDown(self):
        torch.set_default_device(self.device)

    def test_batched_actor_and_gradients_match_scalar(self):
        for name in MODELS:
            with self.subTest(model=name):
                agent = BCAgent(3, 2, config(name))
                obs = torch.randn(3, 5, 3)
                masks = topk(torch.randn(3, 16), 4).requires_grad_()
                batch = agent.actor.forward_batched(obs, masks)
                scalar = torch.stack([agent.actor(o, m) for o, m in zip(obs, masks)])
                torch.testing.assert_close(batch, scalar)
                batch_grads = torch.autograd.grad(batch.square().sum(), masks)[0]
                scalar_grads = torch.autograd.grad(scalar.square().sum(), masks)[0]
                torch.testing.assert_close(batch_grads, scalar_grads)
                self.assertGreater(batch_grads[masks == 0].abs().sum().item(), 0)

    def test_lora_layer_matches_upstream_formula(self):
        cfg = config('lora', lora_bias=True)
        layer = LoRALinear(3, 2, 16, 4, cfg)
        x, mask = torch.randn(2, 5, 3), topk(torch.randn(2, 16), 4)
        projected = torch.einsum('sli,hir->slhr', x, layer.A)
        delta = torch.einsum('slhr,sh,hro->slo', projected, mask, layer.B)
        expected = layer.base(x) + cfg.lora_alpha / 4 * (delta + (mask @ layer.bias_bank)[:, None])
        torch.testing.assert_close(layer(x, mask), expected)

    def test_training_updates_atoms_but_never_backbone(self):
        for name in ('lora', 'bayeslora'):
            for segments in (False, True):
                with self.subTest(model=name, segments=segments):
                    cfg = config(name, segments)
                    agent = BCAgent(3, 2, cfg)
                    weights = {n: p.detach().clone() for n, p in agent.actor.named_parameters()}
                    obs, act = torch.randn(2, 5, 3), torch.randn(2, 5, 2)
                    if segments:
                        masks, _, _, _ = mt10.batched_reconfig(agent, obs, act, cfg, None)
                        mt10.batched_bc_step(agent, obs, act, masks)
                    else:
                        agent.reconfig_actor_mask(obs[0], act[0], True)
                        agent.bc_update(obs[0], act[0], [])
                    for n, p in agent.actor.named_parameters():
                        if '.base.' in n:
                            self.assertFalse(p.requires_grad)
                            torch.testing.assert_close(p, weights[n], rtol=0, atol=0)
                            self.assertIsNone(p.grad)
                    self.assertTrue(any(not torch.equal(p, weights[n]) for n, p in agent.actor.named_parameters() if '.base.' not in n))
                    self.assertFalse(list(agent.inference.state.parameters()))

    def test_cold_evaluation_does_not_change_posterior_or_weights(self):
        for name in MODELS:
            with self.subTest(model=name):
                cfg = config(name)
                agent = BCAgent(3, 2, cfg)
                obs, act = torch.randn(6, 3), torch.randn(6, 2)
                agent.reconfig_actor_mask(obs, act, True)
                state = copy.deepcopy(agent.inference.state_dict())
                weights = copy.deepcopy(agent.actor.state_dict())
                mask, logits = agent.m_a.clone(), agent.m_a_logits.clone()
                found, stats = agent.reconfig_actor_mask(obs, act, False, steps=2)
                self.assertEqual(found.sum(), cfg.ftn_k)
                self.assertTrue(np.isfinite(stats['bc_final']))
                for k, v in state['stream'].items():
                    torch.testing.assert_close(agent.inference.state.state_dict()[k], v, rtol=0, atol=0)
                for k, v in weights.items():
                    torch.testing.assert_close(agent.actor.state_dict()[k], v, rtol=0, atol=0)
                torch.testing.assert_close(mask, agent.m_a)
                torch.testing.assert_close(logits, agent.m_a_logits)

    def test_bayes_kl_prior_decay_sampling_and_best(self):
        previous = torch.randn(2, 16)
        target = torch.randn(2, 16)
        loss = lambda m: (m - target).square().mean(-1)
        self.assertTrue(torch.equal(bernoulli_kl(previous, previous, 1), torch.zeros(2)))
        for sampling in ('gumbel', 'annealste'):
            for take_best in (False, True):
                cfg = config('bayesftn', bayes_sample_type=sampling, bayes_reconfig_take_best=take_best)
                mask, logits, stats = posterior_update(loss, previous, None, cfg)
                torch.testing.assert_close(mask.sum(-1), torch.full((2,), 4.))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue((stats['kl'] >= -1e-7).all())
        cfg = config('bayesftn', bayes_stochastic_mask=False, bayes_reconfig_lr=0,
                     bayes_prior_decay=.5)
        mask, logits, _ = posterior_update(loss, previous, None, cfg)
        torch.testing.assert_close(logits, previous * .5)
        torch.testing.assert_close(mask, topk(logits, 4))
        cfg.bayes_changepoint_prior_decay = 0
        torch.manual_seed(42)
        reset = posterior_update(loss, previous, None, cfg, changepoint=True)
        torch.manual_seed(42)
        other = posterior_update(loss, previous + 100, None, cfg, changepoint=True)
        torch.testing.assert_close(reset[1], other[1])
        cfg.bayes_reconfig_lr = .3
        cfg.bayes_prior_decay = 1
        cfg.bayes_kl_weight = 0
        low = posterior_update(loss, previous, None, cfg)[1]
        cfg.bayes_kl_weight = 100
        high = posterior_update(loss, previous, None, cfg)[1]
        self.assertFalse(torch.allclose(low, high), 'KL must affect multi-step inference')

    def test_reactive_detector_warmup_and_reset(self):
        cfg = config('bayesftn', reconfig_reactive_ema=.5, reconfig_reactive_sd=2)
        agent = BCAgent(3, 2, cfg)
        state = agent.inference.state.select([0], 'cpu')
        self.assertTrue(detect_changepoints(torch.tensor([1.]), state, cfg).item())
        update_detector(state, torch.tensor([1.]), torch.tensor([True]), cfg)
        self.assertFalse(detect_changepoints(torch.tensor([100.]), state, cfg).item())
        update_detector(state, torch.tensor([1.]), torch.tensor([False]), cfg)
        self.assertTrue(detect_changepoints(torch.tensor([100.]), state, cfg).item())
        update_detector(state, torch.tensor([2.]), torch.tensor([True]), cfg)
        self.assertEqual(state['mean'].item(), 2)
        self.assertEqual(state['variance'].item(), 0)
        self.assertEqual(state['since'].item(), 1)

    def test_segment_posterior_identity_and_independent_changepoints(self):
        cfg = config('bayeslora', True, bayes_reconfig_steps=2,
                     bayes_changepoint_reconfig_steps=4, reconfig_reactive_ema=.5)
        agent = BCAgent(3, 2, cfg)
        agent.inference.init_segments(5)
        obs, act = torch.randn(2, 5, 3), torch.randn(2, 5, 2)
        mt10.batched_reconfig(agent, obs, act, cfg, None, segment_ids=[4, 1])
        bank = agent.inference.segment_state
        self.assertEqual(bank.initialized.tolist(), [False, True, False, False, True])
        saved = bank.logits[4].clone()
        mt10.batched_reconfig(agent, obs[:1], act[:1], cfg, None, segment_ids=[1])
        torch.testing.assert_close(bank.logits[4], saved, rtol=0, atol=0)
        self.assertEqual(agent.inference.last_stats['steps'].tolist(), [2])
        # One old segment and one new segment: separate ordinary/reset budgets.
        mt10.batched_reconfig(agent, obs, act, cfg, None, segment_ids=[4, 2])
        self.assertEqual(agent.inference.last_stats['steps'].tolist(), [2, 4])

    def test_batched_posterior_matches_independent_updates(self):
        cfg = config('bayesftn', bayes_stochastic_mask=False, reconfig_overlap_lambda=.3)
        previous, reference = torch.randn(3, 16), topk(torch.randn(3, 16), 4)
        target = torch.randn(3, 16)
        masks, logits, _ = posterior_update(lambda m: (m-target).square().mean(-1), previous, reference, cfg)
        for row in range(3):
            m, l, _ = posterior_update(lambda m: (m-target[row:row+1]).square().mean(-1),
                                       previous[row:row+1], reference[row:row+1], cfg)
            torch.testing.assert_close(masks[row:row+1], m)
            torch.testing.assert_close(logits[row:row+1], l)

    def test_posterior_budgets_support_limits_and_eval_independence(self):
        cfg = config('bayesftn', bayes_reconfig_sample_size=2,
                     bayes_changepoint_reconfig_sample_size=3)
        agent = BCAgent(3, 2, cfg)
        obs, act = torch.randn(7, 3), torch.randn(7, 2)
        with patch.object(agent.actor, 'forward_batched', wraps=agent.actor.forward_batched) as forward:
            agent.reconfig_actor_mask(obs, act, True)
            self.assertIn(3, [call.args[0].shape[1] for call in forward.call_args_list])
            forward.reset_mock()
            agent.reconfig_actor_mask(obs, act, True, steps=2)
            self.assertIn(2, [call.args[0].shape[1] for call in forward.call_args_list])
        self.assertEqual(agent.inference.last_stats['steps'].item(), 2)
        torch.manual_seed(123)
        first = agent.reconfig_actor_mask(obs, act, False)[0]
        agent.inference.state.logits.fill_(1000)
        agent.inference.state.masks.copy_(1-agent.inference.state.masks)
        torch.manual_seed(123)
        second = agent.reconfig_actor_mask(obs, act, False)[0]
        torch.testing.assert_close(first, second)

    def test_lora_fast_warm_match_stored_and_overlap(self):
        cfg = config('lora', lora_reconfig_algorithm='annealste', lora_match_stored=True,
                     lora_match_stored_topk=3, lora_match_stored_after=0, reconfig_overlap_lambda=.2)
        initial = torch.randn(2, 16)
        stored = topk(torch.randn(3, 16), 4)
        target = stored[0].expand(2, -1)
        mask, logits, stats = lora_search(lambda m: (m-target).square().mean(-1), initial,
                                         torch.zeros_like(initial), cfg, 2, initial, stored)
        torch.testing.assert_close(mask, target)
        cfg.lora_reconfig_algorithm = 'fast'
        for mode in ('bump', 'gated', 'seen_logits'):
            cfg.reconfig_overlap_mode = mode
            mask, logits, stats = lora_search(lambda m: (m-target).square().mean(-1), initial,
                                              target, cfg, 2, initial, stored)
            self.assertTrue(torch.isfinite(logits).all())
            torch.testing.assert_close(mask.sum(-1), torch.full((2,), 4.))

    def test_checkpoint_roundtrip_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in MODELS:
                with self.subTest(model=name):
                    cfg = config(name)
                    agent = BCAgent(3, 2, cfg)
                    agent.set_normalizer(torch.zeros(3), torch.ones(3))
                    obs, act = torch.randn(5, 3), torch.randn(5, 2)
                    agent.reconfig_actor_mask(obs, act, True)
                    agent.inference.init_segments(3)
                    agent.inference.segment_state.logits[2].fill_(2)
                    expected = agent.actor(obs, agent.m_a).detach()
                    path = str(Path(tmp) / f'{name}.pt')
                    save_checkpoint(path, agent, cfg, 3, 2, 1)
                    restored, _, _ = load_agent(path, RenderConfig(device='cpu', num_threads=1))
                    torch.testing.assert_close(restored.actor(obs, restored.m_a), expected)
                    torch.testing.assert_close(restored.inference.state.logits, agent.inference.state.logits)
                    torch.testing.assert_close(restored.inference.segment_state.logits, agent.inference.segment_state.logits)
            checkpoint = torch.load(Path(tmp) / 'ftn.pt', weights_only=False)
            checkpoint.pop('inference')
            checkpoint['cfg'].pop('model')
            path = str(Path(tmp) / 'legacy.pt')
            torch.save(checkpoint, path)
            restored, cfg, _ = load_agent(path, RenderConfig(device='cpu', num_threads=1))
            self.assertEqual(cfg.model, 'ftn')

    def test_cli_models_and_changepoint_options_for_both_experiments(self):
        for module, train_name in ((mt50, 'train'), (mt10, 'train_par_segments')):
            with tempfile.TemporaryDirectory() as tmp:
                with patch('sys.argv', ['train.py', '--artifact-root', tmp, '--model', 'BayesLoRA',
                    '--bayes-changepoint-reconfig-steps', '7', '--bayes-prior-decay', '.8',
                    '--no-reconfig-reactive', '--no-lora-bias']), patch.object(module, train_name) as train:
                    module.main()
                cfg = train.call_args.args[0]
                self.assertEqual(cfg.bayes_changepoint_reconfig_steps, 7)
                self.assertEqual(cfg.bayes_prior_decay, .8)
                self.assertFalse(cfg.reconfig_reactive)
                self.assertFalse(cfg.lora_bias)
        with self.assertRaises(ValueError):
            validate_model_config(config('bayesftn', bayes_temperature=0))


if __name__ == '__main__':
    unittest.main()
