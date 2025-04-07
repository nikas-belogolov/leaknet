from pathlib import Path
import os

EPOCHS = 200
BATCH_SIZE = 32
TRAIN_SIZE = 0.75
VAL_SIZE = 0.15

ROOT_DIR = Path(__file__).resolve().parent.parent

FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

DATA_RAW_DIR = "data/raw"
DATA_AUGMENTED_DIR = "data/processed"

MODELS_DIR = os.path.join(ROOT_DIR, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")

DEPLOYMENT_DIR = os.path.join(ROOT_DIR, "deployment")
MODEL_STORE_DIR = os.path.join(DEPLOYMENT_DIR, "model_store")

NORMAL_DATA = os.path.join(ROOT_DIR, DATA_AUGMENTED_DIR, "no_leak")
ANOMALOUS_DATA = os.path.join(ROOT_DIR, DATA_AUGMENTED_DIR, "leak")