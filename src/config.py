"""Paths, column names and hyperparameter grids shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "Train_data.csv"
TEST_CSV = DATA_DIR / "Test_data.csv"  # unlabelled - used by predict.py
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"

TARGET = "class"  # "normal" or "anomaly"
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

SEED = 12          # fixed seed so every run gives the same split and models
TEST_SIZE = 0.20   # 80/20 split -> 5,039-row held-out test set
VAL_SIZE = 0.15    # share of the training set held out to tune the neural network
CV_FOLDS = 5       # k-fold cross-validation for the scikit-learn grid searches

# Neural network ------------------------------------------------------------
NN_EPOCHS = 10

# Untuned starting configuration (the baseline network).
NN_BASELINE = {"lr": 0.001, "hidden1": 20, "hidden2": 10, "batch_size": 64}

# Hyperparameter grid searched for the tuned network.
NN_GRID = {
    "lr": [0.01, 0.001],
    "hidden1": [20, 50],
    "hidden2": [10, 25],
    "batch_size": [32, 64],
}

# Display name -> file stem in models/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Neural Network (tuned)": "neural_network",
}
