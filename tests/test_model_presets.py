"""Portable YAML presets, strict validation, precedence, and run provenance."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from models.preset_config import load_model_yaml
from models.config import model_name
from mt50_continual_learning import train_mt50_bc as mt50
from mt10_segment_representation import train_mt10_transitions_bc_par as mt10

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ('ftn', 'lora', 'bayesftn', 'bayeslora',
           'ftn_reactive_low_noreplay', 'lora_reactive_low_noreplay')


class ModelPresetTests(unittest.TestCase):
    def test_bare_names_ignore_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                Path('lora.yaml').write_text('model: ftn\n')
                for name in ('lora', 'lora.yaml'):
                    overrides, _ = load_model_yaml(name)
                    self.assertEqual(overrides['model'], 'lora')
                overrides, _ = load_model_yaml('./lora.yaml')
                self.assertEqual(overrides['model'], 'ftn')
                for module in (mt10, mt50):
                    cfg = self.run_cli(module, 'bayeslora.yaml', tmp)
                    self.assertEqual(cfg.model, 'bayeslora')
            finally:
                os.chdir(original)

    def run_cli(self, module, path, output, extra=()):
        method = 'train' if module is mt50 else 'train_par_segments'
        with patch('sys.argv', ['train.py', '--artifact-root', output,
                               '--model-config', str(path), *extra]), patch.object(module, method) as train:
            module.main()
        return train.call_args.args[0]

    def test_every_preset_in_both_experiments(self):
        for preset in PRESETS:
            path = ROOT / 'models' / 'presets' / f'{preset}.yaml'
            overrides, source = load_model_yaml(path)
            for module, run_name in ((mt50, 'mt50-continual'), (mt10, 'mt10-segments')):
                with self.subTest(preset=preset, experiment=run_name), tempfile.TemporaryDirectory() as tmp:
                    cfg = self.run_cli(module, path, tmp)
                    self.assertEqual(model_name(cfg), preset.split('_')[0])
                    self.assertEqual(cfg.ftn_hidden_size, overrides['ftn_hidden_size'])
                    self.assertEqual(cfg.ftn_k, overrides['ftn_k'])
                    if 'noreplay' in preset or module is mt10:
                        self.assertFalse(cfg.consolidate)
                        self.assertEqual(cfg.replay_max_chunks, 0)
                    else:
                        self.assertTrue(cfg.consolidate)
                    if 'reactive_low' in preset:
                        self.assertTrue(cfg.reconfig_reactive)
                        self.assertEqual(cfg.reconfig_reactive_sd, 5)
                        self.assertEqual(cfg.reconfig_reactive_ema, .9)
                    run = Path(tmp) / run_name
                    self.assertEqual((run / 'model_config.yaml').read_text(), source)
                    manifest = json.loads((run / 'run.json').read_text())
                    self.assertEqual(manifest['model_config'], 'model_config.yaml')
                    resolved = json.loads((run / 'config.json').read_text())
                    self.assertEqual(resolved['model'], overrides['model'])
                    self.assertEqual(resolved['consolidate'], cfg.consolidate)

    def test_smoke_cli_and_experiment_precedence(self):
        for module in (mt10, mt50):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'custom.yaml'
                path.write_text('model: bayeslora\nlora_rank: 7\nftn_hidden_size: 2304\n'
                                'ftn_k: 256\nftn_square_dim: 48\nconsolidate: true\n'
                                'bayes_changepoint_reconfig_steps: 17\n')
                cfg = self.run_cli(module, path, tmp, ['--smoke', '--lora-rank', '2'])
                self.assertEqual(cfg.model, 'bayeslora')
                self.assertEqual(cfg.lora_rank, 2)  # CLI > YAML
                self.assertEqual(cfg.bayes_changepoint_reconfig_steps, 17)
                self.assertEqual(cfg.ftn_hidden_size, 64)  # smoke > YAML
                self.assertEqual(cfg.ftn_k, 16)
                self.assertEqual(cfg.consolidate, module is mt50)
                cfg = self.run_cli(module, path, tmp, ['--smoke', '--ftn-hidden-size', '128'])
                self.assertEqual(cfg.ftn_hidden_size, 128)  # CLI > smoke

    def test_invalid_yaml_fails_before_writing_artifacts(self):
        invalid = (
            '', '[]', 'lora_rank: 4', 'model: ftn\nmodel: lora',
            'model: ftn\nunknown: 1', 'model: ftn\nftn_k: true',
            'model: ftn\nconsolidate: "false"', 'model: ftn\nftn_k: 4.5',
            'model: ftn\nseed: 3', 'model: ftn\nreconfig_lr: .nan',
            '!!python/object/apply:os.system ["false"]',
            'model: unknown', 'model: bayesftn\nbayes_temperature: 0',
        )
        for source in invalid:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'bad.yaml'
                path.write_text(source)
                output = Path(tmp) / 'runs'
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.run_cli(mt50, path, str(output))
                self.assertFalse(output.exists())

    def test_missing_file_and_optional_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Cannot load model config'):
                load_model_yaml(Path(tmp) / 'missing.yaml')
            path = Path(tmp) / 'config.yaml'
            path.write_text('model: bayesftn\nbayes_reconfig_sample_size: null\n')
            overrides, _ = load_model_yaml(path)
            self.assertIsNone(overrides['bayes_reconfig_sample_size'])


if __name__ == '__main__':
    unittest.main()
