"""Dependency-free smoke tests: python -m unittest discover -s tests."""
import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CLITest(unittest.TestCase):
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "cli.inference_svc", "--help"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        for option in ("--prompt_wav_path", "--target_wav_path", "--rmvpe_path", "--fp16"):
            self.assertIn(option, result.stdout)

    def test_parser(self):
        from cli.inference_svc import build_parser

        args = build_parser().parse_args([
            "--prompt_wav_path", "prompt.wav", "--target_wav_path", "target.wav",
        ])
        self.assertTrue(args.model_path.endswith("model-svc.pt"))
        self.assertIsNone(args.prompt_f0_path)
        self.assertIsNone(args.target_f0_path)

    def test_python_syntax(self):
        for folder in ("cli", "soulxsinger"):
            for path in (ROOT / folder).rglob("*.py"):
                with self.subTest(path=path):
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
