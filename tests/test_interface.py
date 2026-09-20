import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from refineedit.cli import main, read_pairs
from refineedit.config import RefineEditConfig, checkpoint_paths, check_weights


class InterfaceTests(unittest.TestCase):
    def test_defaults(self):
        config = RefineEditConfig()
        self.assertEqual((config.switch_step, config.tau_spatial, config.tau_power),
                         (12, 0.015, 0.12))
        self.assertTrue(config.adaptive_spatial_freezing)
        self.assertEqual(config.bit_lock_steps, 4)

    def test_invalid_configuration(self):
        for parameters in [
            {"switch_step": -1}, {"switch_step": 50}, {"steps": 1},
            {"bit_lock_steps": 0}, {"mask_interval": 0}, {"temperature": 0},
            {"tau_spatial": float("nan")}, {"tau_power": float("inf")},
            {"freeze_multiplier": -1}, {"seed": -1}, {"steps": 4.5},
        ]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                RefineEditConfig(**parameters)

    def test_prompt_formats(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.txt"
            path.write_text('\n["a cat", "a tiger"]\n'
                            '{"source_prompt": "a red car", "editing_prompt": "a blue car"}\n')
            self.assertEqual(read_pairs(path), [["a cat", "a tiger"], ["a red car", "a blue car"]])

    def test_invalid_prompts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.jsonl"
            for data in ["", "{}", '["a cat"]', '["a cat", ""]', '["a cat", 4]']:
                path.write_text(data)
                with self.subTest(data=data), self.assertRaises(ValueError):
                    read_pairs(path)

    def test_dry_run_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()) as stream:
            output = Path(temporary) / "unused"
            main(["--source-prompt", "a cat", "--editing-prompt", "a tiger",
                  "--output-dir", str(output), "--dry-run"])
            self.assertEqual(json.loads(stream.getvalue())["sample_range"], [0, 1])
            self.assertFalse(output.exists())

    def test_individual_sample(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()) as stream:
            prompts = Path(temporary) / "pairs.jsonl"
            prompts.write_text('["a cat", "a tiger"]\n["a dog", "a wolf"]\n')
            main(["--prompts-file", str(prompts), "--start-index", "1", "--end-index", "2",
                  "--output-dir", str(Path(temporary) / "unused"), "--dry-run"])
            self.assertEqual(json.loads(stream.getvalue())["sample_range"], [1, 2])

    def test_missing_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "download_weights.py"):
                check_weights(checkpoint_paths(temporary))

    def test_weight_overrides(self):
        self.assertEqual(checkpoint_paths("unused", "a.pth", "b.ckpt", "text"),
                         (Path("a.pth"), Path("b.ckpt"), Path("text")))
