"""Model definitions: three scikit-learn classifiers and the PyTorch network."""
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.data import build_preprocessor


# --------------------------------------------------------------------------
# scikit-learn models: (pipeline, hyperparameter grid for GridSearchCV)
# --------------------------------------------------------------------------
def sklearn_models():
    seed = config.SEED
    return {
        "Logistic Regression": (
            Pipeline([
                ("prep", build_preprocessor(scale=True)),
                ("clf", LogisticRegression(max_iter=2000)),
            ]),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            Pipeline([
                ("prep", build_preprocessor(scale=False)),
                ("clf", DecisionTreeClassifier(random_state=seed)),
            ]),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 3, 5],
            },
        ),
        "Random Forest": (
            Pipeline([
                ("prep", build_preprocessor(scale=False)),
                ("clf", RandomForestClassifier(random_state=seed, n_jobs=1)),
            ]),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 20],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
    }


# --------------------------------------------------------------------------
# PyTorch feed-forward network (TunableNN in the notebook)
# --------------------------------------------------------------------------
class IntrusionNet(nn.Module):
    """Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid."""

    def __init__(self, input_size, hidden1=20, hidden2=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)).squeeze(-1)


def _tensor(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


def train_nn(X_train, y_train, X_val, y_val, lr=0.001, hidden1=20, hidden2=10,
             batch_size=64, epochs=config.NN_EPOCHS, seed=config.SEED):
    """Train with Adam + binary cross-entropy. Returns (model, loss history)."""
    torch.manual_seed(seed)
    model = IntrusionNet(X_train.shape[1], hidden1, hidden2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tr, y_tr = _tensor(X_train), _tensor(y_train)
    X_v, y_v = _tensor(X_val), _tensor(y_val)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    history = {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        history["train_loss"].append(running / len(X_tr))

        model.eval()
        with torch.no_grad():
            history["val_loss"].append(criterion(model(X_v), y_v).item())
    return model, history


def predict_proba_nn(model, X):
    """Probability that each row is an anomaly."""
    model.eval()
    with torch.no_grad():
        return model(_tensor(X)).numpy()
