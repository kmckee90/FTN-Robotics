"""Named checkpoint routing and display-only global mask consolidation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from bc import BCAgent, checkpoint_masks
from models import mask_to_grid
from rendering import archive_masks, raster_masks, resolve_checkpoint, MaskEMA, compose_frame
from runtime import DEFAULT_RUN_NAMES, named_run
from spatial_permute import spatial_permutation, _geometry
from mt10_segment_representation import train_mt10_transitions_bc_par as mt10
from mt50_continual_learning import train_mt50_bc as mt50


class RenderingTests(unittest.TestCase):
    def test_default_names_and_custom_checkpoint_routing(self):
        for experiment, module, training in (("mt10", mt10, "train_par_segments"), ("mt50", mt50, "train")):
            with tempfile.TemporaryDirectory() as tmp:
                with patch('sys.argv', ['train.py', '--artifact-root', tmp]), patch.object(module, training) as train:
                    module.main()
                run = named_run(tmp, DEFAULT_RUN_NAMES[experiment])
                manifest = json.loads((run / 'run.json').read_text())
                self.assertEqual(manifest['checkpoint'], 'model.pt')
                (run / 'model.pt').touch()
                self.assertEqual(resolve_checkpoint(run, experiment), run / 'model.pt')
                self.assertTrue(train.called)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('sys.argv', ['train.py', '--name', 'whatever0', '--artifact-root', tmp,
                                   '--ckpt-name', 'custom.pt']), patch.object(mt10, 'train_par_segments'):
                mt10.main()
            run = Path(tmp) / 'whatever0'
            (run / 'custom.pt').touch()
            self.assertEqual(resolve_checkpoint(run, 'mt10'), run / 'custom.pt')
            with self.assertRaises(ValueError):
                resolve_checkpoint(run, 'mt50')

    def test_mask_archive_and_block_grid_roundtrip(self):
        cfg = mt50.smoke_config()
        agent = BCAgent(3, 2, cfg)
        agent.stored_m_a = {i: torch.roll(agent.m_a, i) for i in range(2)}
        archive = checkpoint_masks(agent, cfg)
        masks, ids, scope = archive_masks({'mask_archive': archive}, list(cfg.tasks))
        self.assertEqual(ids.tolist(), [0, 1])
        np.testing.assert_array_equal(masks[0], agent.m_a.cpu().numpy())
        # Multiple blocks must use mask_to_grid raster ordering, not vector ordering.
        raw = np.arange(32).reshape(1, 32)
        expected = mask_to_grid(torch.tensor(raw[0]), 4).cpu().numpy().ravel()
        np.testing.assert_array_equal(raster_masks(raw, 4)[0], expected)

    def test_global_permutation_preserves_all_masks_and_overlaps(self):
        masks = (np.random.default_rng(42).random((12, 64)) < .25).astype(float)
        before = masks.copy()
        order = spatial_permutation(masks, (8, 8), sweeps=2)
        np.testing.assert_array_equal(np.sort(order), np.arange(64))
        np.testing.assert_array_equal(masks, before)
        np.testing.assert_array_equal(masks @ masks.T, masks[:, order] @ masks[:, order].T)
        np.testing.assert_array_equal(order, spatial_permutation(masks, (8, 8), sweeps=2))
        left, right, weights = _geometry((8, 8), 3, 2.)
        yy, xx = np.indices((8, 8), dtype=float)
        distances = ((xx - 3.5)**2 + (yy - 3.5)**2).ravel()
        def score(layout):
            moved = masks[:, layout]
            return weights @ (moved[:, left] * moved[:, right]).mean(0) - .4 * distances @ moved.any(0) / distances.max()
        self.assertGreaterEqual(score(order) + 1e-9, score(np.arange(64)))

    def test_two_panel_composition_and_permuted_ema(self):
        mask = torch.zeros(64)
        mask[:8] = 1
        first = MaskEMA(64, 8)
        order = np.arange(64)[::-1].copy()
        other = MaskEMA(64, 8, order)
        first.update(mask)
        other.update(mask)
        np.testing.assert_array_equal(other.image().reshape(-1, 3), first.image().reshape(-1, 3)[order])
        frame = compose_frame(np.zeros((64, 64, 3), dtype=np.uint8), other.image(), 'test', 64)
        self.assertEqual(frame.size, (128, 94))


if __name__ == '__main__':
    unittest.main()
