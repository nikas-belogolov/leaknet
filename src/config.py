from pathlib import Path
import os

EPOCHS = 30
BATCH_SIZE = 32
TRAIN_SIZE = 0.75
VAL_SIZE = 0.15

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = "data/raw"
DATA_AUGMENTED_DIR = "data/processed"

MODELS_DIR = os.path.join(ROOT_DIR, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")


NORMAL_DATA = os.path.join(ROOT_DIR, DATA_AUGMENTED_DIR, "no_leak")
ANOMALOUS_DATA = os.path.join(ROOT_DIR, DATA_AUGMENTED_DIR, "leak")