"""
config.py - single place for every setting in the project.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------- Paths ----------------
DATA_DIR = ROOT / "data"
TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"

PROCESSED_DIR = DATA_DIR / "processed"
WINDOW_DIR = DATA_DIR / "windows"
MODEL_DIR = ROOT / "saved_model"
RESULT_DIR = ROOT / "results"

for d in (PROCESSED_DIR, WINDOW_DIR, MODEL_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- Data ----------------
WINDOW_SIZE = 30          # cycles per input sequence
MAX_RUL = 125             # piece-wise linear RUL cap (training labels)

# ---------------- Training ----------------
BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SIZE = 0.2
SEED = 42

# ---------------- Transformer hyper-parameters ----------------
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.10
LEARNING_RATE = 1e-3
