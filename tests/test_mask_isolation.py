"""Exact inactive-weight isolation, including stale Adam state and changing replay."""
import copy
import os
import unittest
import torch
from bc import BCAgent
from mt10_segment_representation import train_mt10_transitions_bc_par as mt10
from test_models import config, MODELS


def unit_axis(name):
    return 1 if name == 'linear.weight' else 0


def slices(tensor, axis, inactive):
    return tensor.detach().movedim(axis, 0)[inactive]


class MaskIsolationTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.get_default_device()
        torch.set_default_device(os.environ.get('FTN_ISOLATION_DEVICE', 'cpu'))
        torch.manual_seed(42)

    def tearDown(self):
        torch.set_default_device(self.device)

    def masks(self):
        masks = torch.zeros(3, 16)
        for i in range(3):
            masks[i, 4*i:4*(i+1)] = 1
        return masks

    def assert_inactive_unchanged(self, agent, before, active):
        for name, p in agent.actor.named_parameters():
            if not p.requires_grad:
                torch.testing.assert_close(p, before[name], rtol=0, atol=0)
                continue
            axis = unit_axis(name)
            self.assertEqual(p.shape[axis], 16, f'unaccounted shared parameter {name}')
            old = slices(before[name], axis, ~active)
            new = slices(p, axis, ~active)
            self.assertTrue(torch.equal(old, new), f'inactive {name} moved with zero gradients')
            if p.grad is not None:
                self.assertEqual(torch.count_nonzero(slices(p.grad, axis, ~active)).item(), 0)

    def test_inactive_weights_do_not_move_after_mask_switch(self):
        for model in MODELS:
            for segments in (False, True):
                with self.subTest(model=model, segments=segments):
                    agent = BCAgent(3, 2, config(model, segments))
                    masks = self.masks()
                    obs, actions = torch.randn(2, 5, 3), torch.randn(2, 5, 2)
                    for _ in range(3):
                        if segments:
                            mt10.batched_bc_step(agent, obs, actions, masks[:2])
                        else:
                            agent.m_a = masks[0]
                            agent.bc_update(obs[0], actions[0], [])
                    before = copy.deepcopy(agent.actor.state_dict())
                    if segments:
                        active_masks = masks[1:]
                        mt10.batched_bc_step(agent, obs, actions, active_masks)
                        active = active_masks.bool().any(0)
                    else:
                        agent.m_a = masks[1]
                        agent.bc_update(obs[0], actions[0], [])
                        active = masks[1].bool()
                    self.assert_inactive_unchanged(agent, before, active)

    def test_replay_union_and_removed_replay(self):
        for model in MODELS:
            with self.subTest(model=model):
                agent = BCAgent(3, 2, config(model))
                masks = self.masks()
                obs, actions = torch.randn(8, 3), torch.randn(8, 2)
                agent.m_a = masks[0]
                before = copy.deepcopy(agent.actor.state_dict())
                for _ in range(3):
                    agent.bc_update(obs, actions, [(obs, actions, masks[1])])
                self.assert_inactive_unchanged(agent, before, masks[:2].bool().any(0))
                self.assertTrue(any(not torch.equal(slices(p, unit_axis(n), masks[1].bool()),
                    slices(before[n], unit_axis(n), masks[1].bool()))
                    for n, p in agent.actor.named_parameters() if p.requires_grad))
                before = copy.deepcopy(agent.actor.state_dict())
                agent.bc_update(obs, actions, [])
                self.assert_inactive_unchanged(agent, before, masks[0].bool())

    def test_committed_search_resets_weight_adam_but_eval_does_not(self):
        for model in MODELS:
            for segments in (False, True):
                with self.subTest(model=model, segments=segments):
                    cfg = config(model, segments, reconfig_reactive=False)
                    agent = BCAgent(3, 2, cfg)
                    obs, actions = torch.randn(2, 5, 3), torch.randn(2, 5, 2)
                    if segments:
                        masks, _, _, _ = mt10.batched_reconfig(agent, obs, actions, cfg, None)
                        mt10.batched_bc_step(agent, obs, actions, masks)
                    else:
                        agent.bc_update(obs[0], actions[0], [])
                    self.assertTrue(agent.opt.state)
                    saved = copy.deepcopy(agent.opt.state_dict())
                    agent.reconfig_actor_mask(obs[0], actions[0], commit=False)
                    for index, state in saved['state'].items():
                        for key, value in state.items():
                            torch.testing.assert_close(agent.opt.state_dict()['state'][index][key], value)
                    if segments:
                        mt10.batched_reconfig(agent, obs, actions, cfg, None)
                    else:
                        agent.reconfig_actor_mask(obs[0], actions[0], commit=True)
                    self.assertFalse(agent.opt.state)

    def test_nonzero_weight_decay_is_rejected(self):
        from runtime import validate_config
        for model in MODELS:
            cfg = config(model, weight_decay=.1)
            with self.assertRaisesRegex(ValueError, 'weight_decay'):
                validate_config(cfg)
            with self.assertRaisesRegex(ValueError, 'weight_decay'):
                BCAgent(3, 2, cfg)

    def test_raw_weight_gradients_are_zero_outside_active_union(self):
        for model in MODELS:
            with self.subTest(model=model):
                agent = BCAgent(3, 2, config(model))
                owned = {id(p) for p, _ in agent.actor.unit_parameters()}
                trainable = {id(p) for p in agent.actor.parameters() if p.requires_grad}
                self.assertEqual(owned, trainable, 'shared trainable parameter outside unit banks')
                masks = self.masks()[:2]
                obs, actions = torch.randn(2, 5, 3), torch.randn(2, 5, 2)
                (agent.actor.forward_batched(obs, masks)-actions).square().mean().backward()
                active = masks.bool().any(0)
                # Check before the optimizer guard can zero anything.
                for name, p in agent.actor.named_parameters():
                    if p.requires_grad:
                        self.assertEqual(torch.count_nonzero(slices(p.grad, unit_axis(name), ~active)).item(), 0)
                    else:
                        self.assertIsNone(p.grad)

    def test_reconfiguration_leaves_weight_gradients_untouched(self):
        for model in MODELS:
            agent = BCAgent(3, 2, config(model))
            for p in agent.actor.parameters():
                if p.requires_grad:
                    p.grad = torch.randn_like(p)
            before = [None if p.grad is None else p.grad.clone() for p in agent.actor.parameters()]
            agent.reconfig_actor_mask(torch.randn(5, 3), torch.randn(5, 2), commit=True)
            for p, old in zip(agent.actor.parameters(), before):
                if old is None:
                    self.assertIsNone(p.grad)
                else:
                    torch.testing.assert_close(p.grad, old, rtol=0, atol=0)

    def test_reactive_hold_keeps_adam_history(self):
        for model in ('ftn', 'lora'):
            agent = BCAgent(3, 2, config(model, reconfig_reactive=True,
                                        lora_reactive_continuous_updates=False))
            obs, actions = torch.randn(5, 3), torch.randn(5, 2)
            agent.reconfig_actor_mask(obs, actions, commit=True)
            agent.bc_update(obs, actions, [])
            saved = copy.deepcopy(agent.opt.state_dict())
            agent.reconfig_actor_mask(obs, actions, commit=True)
            self.assertEqual(agent.inference.last_stats['steps'].item(), 0)
            for index, state in saved['state'].items():
                for key, value in state.items():
                    torch.testing.assert_close(agent.opt.state_dict()['state'][index][key], value)

    def test_disjoint_mask_predictions_are_unchanged(self):
        for model in MODELS:
            with self.subTest(model=model):
                agent = BCAgent(3, 2, config(model))
                masks = self.masks()
                obs, actions = torch.randn(8, 3), torch.randn(8, 2)
                agent.actor.eval()
                with torch.no_grad():
                    before = agent.actor(obs, masks[1]).clone()
                agent.m_a = masks[0]
                for _ in range(3):
                    agent.bc_update(obs, actions, [])
                with torch.no_grad():
                    after = agent.actor(obs, masks[1])
                self.assertTrue(torch.equal(before, after))


if __name__ == '__main__':
    unittest.main()
