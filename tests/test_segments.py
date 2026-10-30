"""Batch semantics, mask validity, and independent experiment configurations."""
import unittest
import numpy as np
import torch
from bc import BCAgent
from data import segment_windows
from models import anneal_ste_mask_batched
from mt10_segment_representation import train_mt10_transitions_bc_par as segments


class SegmentTests(unittest.TestCase):
    def test_batched_forward_matches_individual_masks(self):
        cfg = segments.smoke_config()
        agent = BCAgent(3, 2, cfg)
        obs = torch.randn(3, 5, 3)
        masks = torch.zeros(3, cfg.ftn_hidden_size)
        for i in range(3):
            masks[i, i:i+cfg.ftn_k] = 1
        expected = torch.stack([agent.actor(o, m) for o, m in zip(obs, masks)])
        torch.testing.assert_close(segments.batched_actor(agent.actor, obs, masks), expected)

    def test_zero_step_search_returns_valid_masks_and_finite_losses(self):
        masks, logits, initial, final = anneal_ste_mask_batched(
            16, 4, lambda m: m.square().mean(-1), 0, .3, 3)
        torch.testing.assert_close(masks.sum(-1), torch.full((3,), 4.))
        self.assertTrue(np.isfinite(initial) and np.isfinite(final))

    def test_windows_and_batches_preserve_coverage(self):
        self.assertEqual(segment_windows(10, 4, 4), [(0, 4), (4, 8), (6, 10)])
        windows = [(0, 0, 4), (0, 2, 6), (1, 0, 4), (1, 4, 8), (2, 0, 4)]
        for mode in ('random', 'trajectory', 'contiguous'):
            batches = segments.build_batches(windows, 2, mode)
            self.assertEqual(sorted(i for batch in batches for i in batch), list(range(5)))
            self.assertEqual(len(batches), segments._n_batches(windows, 2, mode))
            if mode == 'trajectory':
                for batch in batches:
                    self.assertEqual(len({windows[i][0] for i in batch}), 1)


if __name__ == '__main__':
    unittest.main()
