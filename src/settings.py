import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).expanduser()
if not DATA_DIR.is_absolute():
    DATA_DIR = PROJECT_ROOT / DATA_DIR

OUTPUT_DIR = DATA_DIR / "outputs"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
DATASET_DIR = DATA_DIR / "dataset"
