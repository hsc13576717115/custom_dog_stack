from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_conversion_script_is_compilable() -> None:
    source = Path(__file__).resolve().parents[1] / "scripts" / "convert_distillation_student_checkpoint.py"
    compile(source.read_text(encoding="utf-8"), str(source), "exec")
