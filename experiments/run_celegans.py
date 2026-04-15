"""
WiringMatters — Stage 1 Benchmark: C. elegans Topology
=======================================================

Run the benchmark on standard ML tasks across up to four conditions:
  1. Dense baseline          — fully connected MLP / RNN
  2. Uniform sparse baseline — randomly sparse (matched density)
  3. Bio-topological         — masked by C. elegans connectome topology
  4. Magnitude pruned        — one-shot magnitude pruning (MLP tasks only)

Usage:
    python experiments/run_celegans.py
    python experiments/run_celegans.py --task all --hidden 448 --epochs 50
    python experiments/run_celegans.py --task all --epochs 100 --seeds 5
    python experiments/run_celegans.py --task sequential_mnist --epochs 30
    python experiments/run_celegans.py --task all --output results/

Output:
    results/celegans_benchmark_<timestamp>.json with all metrics, seeds, masks.
    With --seeds > 1: includes per-seed results and mean ± std aggregation.

Scientific conventions (from whitepaper Section 6):
    - Seeds: either a single fixed seed (default 42) or N consecutive seeds
      starting at 42 (e.g., --seeds 5 runs seeds 42, 43, 44, 45, 46)
    - The bio mask is deterministic (derived from the connectome, no randomness)
    - The uniform sparse mask is re-sampled for each seed for a fair comparison
    - All base conditions always present; magnitude_pruned added for MLP tasks
    - Parameter counts matched across conditions
    - Results logged to structured JSON for reproducibility

Tasks:
    MLP tasks (4 conditions incl. magnitude_pruned):
        digits            — Digits 8x8 image classification (sklearn, 10 classes)
        housing           — Diabetes regression (sklearn, no download)
        moons             — Two Moons 2D classification (sklearn)
    Sequential tasks (3 conditions, BioRNN):
        seq_digits        — Digits read row-by-row: 8 timesteps × 8 features
        sequential_mnist  — MNIST read row-by-row: 28 timesteps × 28 features
                            (requires torchvision + internet on first run)
    all                   — All tasks above except sequential_mnist (no download)
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import wiringmatters as wm

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")

# Seeds (fixed for reproducibility — whitepaper Section 6)
# Default single seed. Multi-seed runs use seeds [BASE_SEED, BASE_SEED+1, ...]
# to maintain reproducibility while estimating variance.

BASE_SEED = 42
SEED = BASE_SEED  # backwards-compatible alias used in dataset splitting
torch.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)

# Task classification
# Used by run_benchmark to dispatch to the right experiment function.
MLP_TASKS = {"digits", "housing", "moons"}
SEQUENTIAL_TASKS = {"seq_digits", "sequential_mnist"}
ALL_DEFAULT_TASKS = ["digits", "housing", "moons", "seq_digits"]


# Dataset loaders

def load_task(task: str) -> dict:
    """
    Load a standard ML benchmark dataset.

    Args:
        task: One of "digits", "housing", "moons" (MLP tasks),
              "seq_digits", "sequential_mnist" (sequential RNN tasks).

    Returns:
        dict with keys: X_train, X_test, y_train, y_test, input_dim,
                        output_dim, task_type, name.
        Sequential tasks also include: seq_len (number of timesteps).
        task_type is "classification", "regression", or "sequence_classification".
    """
    log.info(f"Loading dataset: {task}")

    if task == "digits":
        data = load_digits()
        X, y = data.data.astype(np.float32), data.target.astype(np.int64)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "input_dim": 64, "output_dim": 10,
            "task_type": "classification",
            "name": "Digits (8x8 classification)",
        }

    elif task == "housing":
        # Uses sklearn's built-in Diabetes dataset (442 samples, 10 features,
        # continuous target) — no network download required, works on macOS
        # Python 3.14+ without SSL certificate setup.
        data = load_diabetes()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train.reshape(-1, 1), "y_test": y_test.reshape(-1, 1),
            "input_dim": 10, "output_dim": 1,
            "task_type": "regression",
            "name": "Diabetes (regression)",
        }

    elif task == "moons":
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=2000, noise=0.2, random_state=SEED)
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "input_dim": 2, "output_dim": 2,
            "task_type": "classification",
            "name": "Two Moons (2D classification)",
        }

    elif task == "seq_digits":
        # Sequential Digits: each 8×8 image is read as 8 timesteps of 8 features
        # (one row per timestep). Tests whether temporal processing of biological
        # structure helps on sequential classification without requiring any downloads.
        data = load_digits()
        X_flat, y = data.data.astype(np.float32), data.target.astype(np.int64)
        # Reshape each 64-dim sample into (8, 8) = (timesteps, features)
        X_seq = X_flat.reshape(-1, 8, 8)  # (N, T=8, F=8)

        # Normalise per-feature across training set
        X_train_seq, X_test_seq, y_train, y_test = train_test_split(
            X_seq, y, test_size=0.2, random_state=SEED, stratify=y
        )
        # Fit scaler on flattened train, apply back in sequence shape
        scaler = StandardScaler()
        n_train, T, F = X_train_seq.shape
        X_train_2d = X_train_seq.reshape(n_train, T * F)
        X_test_2d = X_test_seq.reshape(len(X_test_seq), T * F)
        X_train_2d = scaler.fit_transform(X_train_2d).astype(np.float32)
        X_test_2d = scaler.transform(X_test_2d).astype(np.float32)
        X_train_seq = X_train_2d.reshape(n_train, T, F)
        X_test_seq = X_test_2d.reshape(len(X_test_seq), T, F)

        return {
            "X_train": X_train_seq, "X_test": X_test_seq,
            "y_train": y_train, "y_test": y_test,
            "input_dim": 8, "output_dim": 10, "seq_len": 8,
            "task_type": "sequence_classification",
            "name": "Sequential Digits (8 timesteps × 8 features)",
        }

    elif task == "sequential_mnist":
        # Sequential MNIST: each 28×28 image is read as 28 timesteps of 28 features
        # (one pixel row per timestep). Canonical benchmark for RNN memory.
        # Requires torchvision and internet access on first run.
        try:
            import torchvision
            import torchvision.transforms as transforms
        except ImportError:
            raise ImportError(
                "torchvision is required for sequential_mnist. "
                "Install with: pip install torchvision"
            )

        data_dir = Path(__file__).parent.parent / "data" / "mnist"
        data_dir.mkdir(parents=True, exist_ok=True)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        try:
            train_ds = torchvision.datasets.MNIST(
                str(data_dir), train=True, download=True, transform=transform
            )
            test_ds = torchvision.datasets.MNIST(
                str(data_dir), train=False, download=True, transform=transform
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download MNIST: {e}\n"
                "On macOS, run: /Applications/Python*/Install Certificates.command\n"
                "Or use --task seq_digits for a no-download sequential alternative."
            )

        # Convert to numpy arrays for uniform interface
        # Each image (1, 28, 28) → sequence (28, 28)
        rng = np.random.default_rng(SEED)
        train_idx = rng.permutation(len(train_ds))[:10000]  # 10k subset for speed
        test_idx = rng.permutation(len(test_ds))[:2000]

        def _extract(ds, idx):
            imgs = np.stack([ds[i][0].numpy().squeeze() for i in idx])  # (N, 28, 28)
            labels = np.array([ds[i][1] for i in idx], dtype=np.int64)
            return imgs, labels

        X_train_seq, y_train = _extract(train_ds, train_idx)
        X_test_seq, y_test = _extract(test_ds, test_idx)

        return {
            "X_train": X_train_seq.astype(np.float32),
            "X_test": X_test_seq.astype(np.float32),
            "y_train": y_train, "y_test": y_test,
            "input_dim": 28, "output_dim": 10, "seq_len": 28,
            "task_type": "sequence_classification",
            "name": "Sequential MNIST (28 timesteps × 28 features, 10k/2k subset)",
        }

    else:
        raise ValueError(
            f"Unknown task: {task}. "
            f"Choose from: digits, housing, moons, seq_digits, sequential_mnist"
        )


def to_dataloaders(dataset: dict, batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """Convert numpy arrays to PyTorch DataLoaders (MLP tasks)."""
    X_train = torch.tensor(dataset["X_train"])
    y_train = torch.tensor(dataset["y_train"])
    X_test = torch.tensor(dataset["X_test"])
    y_test = torch.tensor(dataset["y_test"])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def to_rnn_dataloaders(dataset: dict, batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """Convert numpy arrays to PyTorch DataLoaders for sequential tasks.

    X tensors have shape (N, T, F) — batch_first convention used by BioRNN.
    """
    X_train = torch.tensor(dataset["X_train"])  # (N, T, F)
    y_train = torch.tensor(dataset["y_train"])  # (N,) for classification
    X_test = torch.tensor(dataset["X_test"])
    y_test = torch.tensor(dataset["y_test"])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


# Connectome mask preparation

def prepare_masks(
    hidden_size: int,
    adjacency: np.ndarray,
    seed: int = BASE_SEED,
) -> dict:
    """
    Build the three base experimental masks for a hidden layer of given size.

    Args:
        hidden_size: Target hidden layer dimension.
        adjacency:   Binary adjacency matrix from the C. elegans connectome.
        seed:        Random seed for the uniform sparse mask. The bio mask is
                     deterministic (derived from the connectome) and is never
                     affected by this seed.

    Returns:
        dict with keys "dense", "uniform_sparse", "bio_topological" mapping to
        torch.Tensor masks (or None for dense), plus "density" (float).

    Mask scaling strategy (whitepaper Section 6):
        The C. elegans connectome dataset has 448 nodes (302 somatic neurons
        + 20 pharyngeal + ~126 muscle/non-neuronal cells). The default
        hidden_size is 448, which uses the full adjacency matrix without any
        modification. If a different size is needed: when target > connectome
        size, the adjacency is tiled with wrap-around; when target < connectome
        size, the top-left submatrix is used. This strategy is kept constant
        across all experiments for reproducibility.
    """
    bio = wm.bio_mask(adjacency, target_shape=(hidden_size, hidden_size))
    density = wm.mask_density(bio)  # returns float directly, no .item() needed
    log.info(f"Bio mask density: {density:.4f} ({density*100:.2f}% connections active)")

    uniform = wm.uniform_sparse_mask(
        target_shape=(hidden_size, hidden_size),
        density=density,
        seed=seed,
    )
    dense = wm.dense_mask(target_shape=(hidden_size, hidden_size))

    return {
        "dense": dense,
        "uniform_sparse": uniform,
        "bio_topological": bio,
        "density": density,
    }


# Training loop (MLP)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    task_type: str,
) -> float:
    """Train for one epoch. Returns mean training loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    task_type: str,
) -> dict:
    """Evaluate model. Returns loss and accuracy (classification) or MSE (regression)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item() * len(X_batch)
        total += len(X_batch)

        if task_type == "classification":
            correct += (pred.argmax(dim=1) == y_batch).sum().item()

    metrics = {"loss": total_loss / total}
    if task_type == "classification":
        metrics["accuracy"] = correct / total
    else:
        metrics["mse"] = total_loss / total

    return metrics


def run_experiment(
    condition_name: str,
    mask: torch.Tensor | None,
    dataset: dict,
    layer_sizes: list[int],
    epochs: int,
    lr: float = 1e-3,
    seed: int = BASE_SEED,
    return_model: bool = False,
) -> dict | tuple[dict, nn.Module]:
    """
    Run one experimental condition for MLP tasks.

    Args:
        condition_name: Label for logging and results.
        mask:           The connectivity mask for hidden layers (or None for dense).
        dataset:        Dataset dict from load_task().
        layer_sizes:    Full list [input_dim, hidden..., output_dim].
        epochs:         Number of training epochs.
        lr:             Learning rate.
        seed:           Random seed for model weight initialisation.
        return_model:   If True, return (result_dict, trained_model) instead of just result_dict.

    Returns:
        dict (or (dict, model) if return_model=True) with condition, metrics, etc.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Condition: {condition_name.upper()}")
    log.info(f"Architecture: {layer_sizes}")

    task_type = dataset["task_type"]

    # Build masks list for BioMLP: apply square mask to hidden↔hidden layers only.
    num_layers = len(layer_sizes) - 1
    if mask is None:
        masks = None
    else:
        masks = []
        for i in range(num_layers):
            in_f, out_f = layer_sizes[i], layer_sizes[i + 1]
            if in_f == out_f:
                masks.append(mask)
            else:
                masks.append(None)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = wm.BioMLP(layer_sizes=layer_sizes, masks=masks, activation="relu")

    total_params = sum(p.numel() for p in model.parameters())
    active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {total_params:,} total, {active_params:,} trainable")

    criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader, test_loader = to_dataloaders(dataset)

    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, task_type)
        test_metrics = evaluate(model, test_loader, criterion, task_type)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            **{f"test_{k}": round(v, 6) for k, v in test_metrics.items()},
        }
        history.append(epoch_record)

        if epoch % 10 == 0 or epoch == 1:
            if task_type == "classification":
                log.info(
                    f"  Epoch {epoch:>3}/{epochs}  "
                    f"train_loss={train_loss:.4f}  "
                    f"test_acc={test_metrics['accuracy']:.4f}"
                )
            else:
                log.info(
                    f"  Epoch {epoch:>3}/{epochs}  "
                    f"train_loss={train_loss:.4f}  "
                    f"test_mse={test_metrics['mse']:.4f}"
                )

    elapsed = time.time() - t0
    final_train = evaluate(model, train_loader, criterion, task_type)
    final_test = evaluate(model, test_loader, criterion, task_type)
    generalization_gap = final_train["loss"] - final_test["loss"]

    log.info(f"  Completed in {elapsed:.1f}s — generalization gap: {generalization_gap:+.4f}")

    result = {
        "condition": condition_name,
        "layer_sizes": layer_sizes,
        "total_params": total_params,
        "active_params": active_params,
        "epochs": epochs,
        "lr": lr,
        "history": history,
        "final_train": final_train,
        "final_test": final_test,
        "generalization_gap": round(generalization_gap, 6),
        "wall_time_seconds": round(elapsed, 2),
    }

    if return_model:
        return result, model
    return result


def _extract_magnitude_mask(
    model: nn.Module,
    layer_sizes: list[int],
    density: float,
) -> torch.Tensor:
    """
    Extract a magnitude-based mask from a trained BioMLP.

    Finds the first square hidden-hidden layer (where in_features == out_features)
    and applies magnitude_mask at the given density. This is used to build the
    "magnitude pruned" condition: keep the top-k% weights by |W| from the dense
    model, then retrain from scratch with that fixed topology.

    Args:
        model:       Trained BioMLP model.
        layer_sizes: Layer size list used to build the model.
        density:     Fraction of weights to keep (matched to bio mask density).

    Returns:
        Binary torch.Tensor mask for the first square hidden layer.
    """
    for i, layer in enumerate(model.layers):
        in_f = layer_sizes[i]
        out_f = layer_sizes[i + 1]
        if in_f == out_f:
            # Extract effective weight (W * M for masked layers, W for dense)
            weight = layer.weight.detach()
            return wm.magnitude_mask(weight, density=density)

    raise RuntimeError(
        f"No square hidden layer found in model with layer_sizes={layer_sizes}"
    )


# Training loop (BioRNN sequential tasks)

def train_one_epoch_rnn(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Train RNN for one epoch on a sequence classification task."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        # X_batch: (batch, T, F)
        optimizer.zero_grad()
        pred = model.get_sequence_output(X_batch)  # (batch, output_size)
        loss = criterion(pred, y_batch)
        loss.backward()
        # Gradient clipping to stabilise RNN training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_rnn(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> dict:
    """Evaluate RNN on sequence classification. Returns loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        pred = model.get_sequence_output(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item() * len(X_batch)
        total += len(X_batch)
        correct += (pred.argmax(dim=1) == y_batch).sum().item()

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def run_rnn_experiment(
    condition_name: str,
    mask_hh: torch.Tensor | None,
    dataset: dict,
    hidden_size: int,
    epochs: int,
    lr: float = 1e-3,
    seed: int = BASE_SEED,
) -> dict:
    """
    Run one experimental condition for sequential tasks using BioRNN.

    The bio/uniform topology is applied to the hidden→hidden recurrent weight (mask_hh).
    The input→hidden projection (mask_ih) is always dense (None) because it maps
    between the input feature space and the hidden space — these have different
    sizes so the square connectome mask cannot be applied there.

    Args:
        condition_name: Label for logging and results.
        mask_hh:        Recurrent connectivity mask (hidden×hidden) or None for dense.
        dataset:        Dataset dict from load_task() with task_type "sequence_classification".
        hidden_size:    Size of the recurrent hidden layer.
        epochs:         Number of training epochs.
        lr:             Learning rate.
        seed:           Random seed for model weight initialisation.

    Returns:
        dict with condition, metrics per epoch, final train/test metrics, param count.
    """
    log.info(f"\n{'='*60}")
    log.info(f"RNN Condition: {condition_name.upper()}")

    input_size = dataset["input_dim"]
    output_size = dataset["output_dim"]
    seq_len = dataset["seq_len"]
    log.info(
        f"Architecture: BioRNN(input={input_size}, hidden={hidden_size}, "
        f"output={output_size}, seq_len={seq_len})"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Input→hidden mask: always dense (rectangular, so the square connectome
    # mask cannot apply here). Shape must be (hidden_size, input_size).
    mask_ih_dense = wm.dense_mask(target_shape=(hidden_size, input_size))

    # Recurrent hidden→hidden mask: applies the connectome topology (or dense).
    # Shape must be (hidden_size, hidden_size).
    if mask_hh is None:
        mask_hh_actual = wm.dense_mask(target_shape=(hidden_size, hidden_size))
    else:
        mask_hh_actual = mask_hh

    model = wm.BioRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        mask_ih=mask_ih_dense,  # always dense input projection
        mask_hh=mask_hh_actual, # topology constraint on recurrent connections
        batch_first=True,       # DataLoader gives (batch, T, F)
    )

    total_params = sum(p.numel() for p in model.parameters())
    active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {total_params:,} total, {active_params:,} trainable")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader, test_loader = to_rnn_dataloaders(dataset)

    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_rnn(model, train_loader, optimizer, criterion)
        test_metrics = evaluate_rnn(model, test_loader, criterion)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "test_loss": round(test_metrics["loss"], 6),
            "test_accuracy": round(test_metrics["accuracy"], 6),
        }
        history.append(epoch_record)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"  Epoch {epoch:>3}/{epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"test_acc={test_metrics['accuracy']:.4f}"
            )

    elapsed = time.time() - t0
    final_train = evaluate_rnn(model, train_loader, criterion)
    final_test = evaluate_rnn(model, test_loader, criterion)
    generalization_gap = final_train["loss"] - final_test["loss"]

    log.info(f"  Completed in {elapsed:.1f}s — generalization gap: {generalization_gap:+.4f}")

    return {
        "condition": condition_name,
        "model_type": "BioRNN",
        "hidden_size": hidden_size,
        "input_size": input_size,
        "output_size": output_size,
        "seq_len": seq_len,
        "total_params": total_params,
        "active_params": active_params,
        "epochs": epochs,
        "lr": lr,
        "history": history,
        "final_train": final_train,
        "final_test": final_test,
        "generalization_gap": round(generalization_gap, 6),
        "wall_time_seconds": round(elapsed, 2),
    }


# Aggregation helpers

def _aggregate_seeds(per_seed: dict) -> dict:
    """
    Compute mean ± std across seeds for every scalar metric in final_test,
    final_train, and generalization_gap.

    Args:
        per_seed: {seed_str: run_result_dict, ...}

    Returns:
        dict with "mean" and "std" for each key found in final_test / final_train.
    """
    runs = list(per_seed.values())

    def stats(values: list[float]) -> dict:
        arr = np.array(values, dtype=np.float64)
        return {"mean": round(float(arr.mean()), 6), "std": round(float(arr.std()), 6)}

    agg: dict = {}

    for split in ("final_test", "final_train"):
        agg[split] = {}
        for key in runs[0][split]:
            agg[split][key] = stats([r[split][key] for r in runs])

    agg["generalization_gap"] = stats([r["generalization_gap"] for r in runs])
    return agg


# Main benchmark orchestrator

def run_benchmark(
    tasks: list[str],
    hidden_size: int,
    epochs: int,
    output_dir: Path,
    seeds: list[int],
) -> None:
    """
    Run the full benchmark on one or more tasks and seeds.

    MLP tasks (digits, housing, moons) run 4 conditions:
        dense, uniform_sparse, bio_topological, magnitude_pruned

    Sequential tasks (seq_digits, sequential_mnist) run 3 conditions:
        dense, uniform_sparse, bio_topological
    (magnitude_pruned is not run for RNN tasks in Stage 1)

    With a single seed this is identical to the original behaviour.
    With multiple seeds every condition is run once per seed and results
    are aggregated (mean ± std) so variance can be reported.

    Follows whitepaper conventions:
      - Always all base conditions
      - Bio mask is deterministic (connectome-derived, seed-independent)
      - Uniform sparse mask is re-sampled per seed for a fair variance estimate
      - Magnitude mask extracted from the trained dense model (per seed)
      - Saves mask arrays for exact reproducibility
      - Structured JSON output
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    multi_seed = len(seeds) > 1

    # Load C. elegans connectome (once — shared across all seeds and tasks)
    log.info("Loading C. elegans connectome...")
    G, adjacency = wm.load_celegans(synapse_type="chemical", as_matrix=True)
    log.info(f"Connectome: {G.number_of_nodes()} neurons, {G.number_of_edges()} synapses")

    # Topological summary (connectome-level, seed-independent)
    log.info("Computing topological summary...")
    topo = wm.topological_summary(G)
    log.info(f"  Clustering: {topo['clustering_coefficient']:.4f}")
    log.info(f"  Modularity: {topo['modularity']:.4f}")

    # Bio mask — deterministic, built once
    bio_mask_tensor = wm.bio_mask(adjacency, target_shape=(hidden_size, hidden_size))
    bio_density = wm.mask_density(bio_mask_tensor)
    log.info(f"Bio mask density: {bio_density:.4f}")

    # Save the bio mask (same for all seeds)
    mask_path = output_dir / f"masks_{timestamp}.npz"
    np.savez(
        mask_path,
        bio_topological=bio_mask_tensor.numpy(),
        dense=wm.dense_mask(target_shape=(hidden_size, hidden_size)).numpy(),
    )
    log.info(f"Bio mask saved to {mask_path}")

    all_results: dict = {
        "meta": {
            "timestamp": timestamp,
            "seeds": seeds,
            "hidden_size": hidden_size,
            "epochs": epochs,
            "connectome": "C. elegans (chemical synapses)",
            "connectome_nodes": G.number_of_nodes(),
            "connectome_edges": G.number_of_edges(),
            "bio_mask_density": bio_density,
            "mask_scaling_strategy": (
                f"hidden_size={hidden_size} vs connectome=448: "
                + ("full matrix (no modification)" if hidden_size == 448
                   else "top-left crop" if hidden_size < 448
                   else "tile with wrap-around")
            ),
            "mask_file": str(mask_path),
        },
        "topology": topo,
        "tasks": {},
    }

    for task_name in tasks:
        log.info(f"\n{'#'*60}")
        log.info(f"TASK: {task_name}")
        log.info(f"{'#'*60}")

        is_sequential = task_name in SEQUENTIAL_TASKS
        dataset = load_task(task_name)

        if is_sequential:
            _run_sequential_task(
                task_name=task_name,
                dataset=dataset,
                hidden_size=hidden_size,
                epochs=epochs,
                seeds=seeds,
                multi_seed=multi_seed,
                bio_mask_tensor=bio_mask_tensor,
                bio_density=bio_density,
                all_results=all_results,
            )
        else:
            _run_mlp_task(
                task_name=task_name,
                dataset=dataset,
                hidden_size=hidden_size,
                epochs=epochs,
                seeds=seeds,
                multi_seed=multi_seed,
                bio_mask_tensor=bio_mask_tensor,
                bio_density=bio_density,
                all_results=all_results,
            )

    # Write JSON results
    result_path = output_dir / f"celegans_benchmark_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"Results saved to: {result_path}")
    log.info(f"{'='*60}")

    _print_summary(all_results)


def _run_mlp_task(
    task_name: str,
    dataset: dict,
    hidden_size: int,
    epochs: int,
    seeds: list[int],
    multi_seed: bool,
    bio_mask_tensor: torch.Tensor,
    bio_density: float,
    all_results: dict,
) -> None:
    """
    Run MLP benchmark for one task across all seeds.

    Four conditions:
        dense, uniform_sparse, bio_topological, magnitude_pruned
    """
    input_dim = dataset["input_dim"]
    output_dim = dataset["output_dim"]
    layer_sizes = [input_dim, hidden_size, hidden_size, output_dim]

    task_results: dict = {
        "dataset": dataset["name"],
        "input_dim": input_dim,
        "output_dim": output_dim,
        "task_type": dataset["task_type"],
        "n_train": len(dataset["X_train"]),
        "n_test": len(dataset["X_test"]),
        "model_type": "BioMLP",
        "conditions": {
            "dense":            {"per_seed": {}},
            "uniform_sparse":   {"per_seed": {}},
            "bio_topological":  {"per_seed": {}},
            "magnitude_pruned": {"per_seed": {}},
        },
    }

    for seed in seeds:
        log.info(f"\n--- Seed {seed} ---")

        # Uniform sparse mask re-sampled per seed
        uniform_mask = wm.uniform_sparse_mask(
            target_shape=(hidden_size, hidden_size),
            density=bio_density,
            seed=seed,
        )

        # Dense condition — run first, keep trained model for magnitude mask
        dense_result, dense_model = run_experiment(
            condition_name="dense",
            mask=None,
            dataset=dataset,
            layer_sizes=layer_sizes,
            epochs=epochs,
            seed=seed,
            return_model=True,
        )
        task_results["conditions"]["dense"]["per_seed"][str(seed)] = dense_result

        # Magnitude pruned: extract mask from trained dense model, retrain from scratch
        log.info("Extracting magnitude mask from trained dense model...")
        try:
            mag_mask = _extract_magnitude_mask(dense_model, layer_sizes, bio_density)
            log.info(
                f"Magnitude mask density: {wm.mask_density(mag_mask):.4f} "
                f"(target: {bio_density:.4f})"
            )
        except RuntimeError as e:
            log.warning(f"Could not extract magnitude mask: {e}. Skipping condition.")
            mag_mask = None

        magnitude_result = run_experiment(
            condition_name="magnitude_pruned",
            mask=mag_mask,
            dataset=dataset,
            layer_sizes=layer_sizes,
            epochs=epochs,
            seed=seed,
        )
        task_results["conditions"]["magnitude_pruned"]["per_seed"][str(seed)] = magnitude_result

        # Uniform sparse and bio-topological conditions
        for condition, mask in [
            ("uniform_sparse",  uniform_mask),
            ("bio_topological", bio_mask_tensor),
        ]:
            result = run_experiment(
                condition_name=condition,
                mask=mask,
                dataset=dataset,
                layer_sizes=layer_sizes,
                epochs=epochs,
                seed=seed,
            )
            task_results["conditions"][condition]["per_seed"][str(seed)] = result

    # Aggregate across seeds
    for condition in ("dense", "uniform_sparse", "bio_topological", "magnitude_pruned"):
        per_seed = task_results["conditions"][condition]["per_seed"]
        task_results["conditions"][condition]["aggregate"] = _aggregate_seeds(per_seed)

        if not multi_seed:
            sole = list(per_seed.values())[0]
            task_results["conditions"][condition].update({
                k: v for k, v in sole.items()
                if k not in ("per_seed", "aggregate")
            })

    all_results["tasks"][task_name] = task_results


def _run_sequential_task(
    task_name: str,
    dataset: dict,
    hidden_size: int,
    epochs: int,
    seeds: list[int],
    multi_seed: bool,
    bio_mask_tensor: torch.Tensor,
    bio_density: float,
    all_results: dict,
) -> None:
    """
    Run RNN benchmark for one sequential task across all seeds.

    Three conditions (magnitude_pruned is not applicable for RNN):
        dense, uniform_sparse, bio_topological
    """
    task_results: dict = {
        "dataset": dataset["name"],
        "input_dim": dataset["input_dim"],
        "output_dim": dataset["output_dim"],
        "seq_len": dataset["seq_len"],
        "task_type": dataset["task_type"],
        "n_train": len(dataset["X_train"]),
        "n_test": len(dataset["X_test"]),
        "model_type": "BioRNN",
        "conditions": {
            "dense":           {"per_seed": {}},
            "uniform_sparse":  {"per_seed": {}},
            "bio_topological": {"per_seed": {}},
        },
    }

    for seed in seeds:
        log.info(f"\n--- Seed {seed} ---")

        # Uniform sparse mask re-sampled per seed
        uniform_mask = wm.uniform_sparse_mask(
            target_shape=(hidden_size, hidden_size),
            density=bio_density,
            seed=seed,
        )

        for condition, mask_hh in [
            ("dense",           None),
            ("uniform_sparse",  uniform_mask),
            ("bio_topological", bio_mask_tensor),
        ]:
            result = run_rnn_experiment(
                condition_name=condition,
                mask_hh=mask_hh,
                dataset=dataset,
                hidden_size=hidden_size,
                epochs=epochs,
                seed=seed,
            )
            task_results["conditions"][condition]["per_seed"][str(seed)] = result

    # Aggregate across seeds
    for condition in ("dense", "uniform_sparse", "bio_topological"):
        per_seed = task_results["conditions"][condition]["per_seed"]
        task_results["conditions"][condition]["aggregate"] = _aggregate_seeds(per_seed)

        if not multi_seed:
            sole = list(per_seed.values())[0]
            task_results["conditions"][condition].update({
                k: v for k, v in sole.items()
                if k not in ("per_seed", "aggregate")
            })

    all_results["tasks"][task_name] = task_results


def _print_summary(results: dict) -> None:
    """Print a formatted summary table of benchmark results."""
    seeds = results["meta"]["seeds"]
    multi_seed = len(seeds) > 1

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY — WiringMatters Stage 1")
    print("="*80)

    if multi_seed:
        print(f"{'Task':<16} {'Model':<7} {'Condition':<20} {'Test metric (mean ± std)':<30} {'Gen. gap'}")
        print("-"*80)
    else:
        print(f"{'Task':<16} {'Model':<7} {'Condition':<20} {'Test metric':<22} {'Gen. gap'}")
        print("-"*80)

    for task_name, task_data in results["tasks"].items():
        task_type = task_data["task_type"]
        model_type = task_data.get("model_type", "BioMLP")
        is_rnn = model_type == "BioRNN"

        metric_key = "accuracy" if task_type in ("classification", "sequence_classification") else "mse"
        metric_label = "acc" if metric_key == "accuracy" else "mse"

        for cond_name, cond_data in task_data["conditions"].items():
            agg = cond_data["aggregate"]
            m = agg["final_test"][metric_key]
            g = agg["generalization_gap"]

            if multi_seed:
                metric_str = f"{metric_label}={m['mean']:.4f} ± {m['std']:.4f}"
                gap_str = f"{g['mean']:+.4f} ± {g['std']:.4f}"
            else:
                metric_str = f"{metric_label}={m['mean']:.4f}"
                gap_str = f"{g['mean']:+.4f}"

            short_model = "RNN" if is_rnn else "MLP"
            print(f"{task_name:<16} {short_model:<7} {cond_name:<20} {metric_str:<30} {gap_str}")
        print()

    print("="*80)
    seed_str = str(seeds[0]) if not multi_seed else f"{seeds[0]}–{seeds[-1]} ({len(seeds)} seeds)"
    print(f"Seeds:            {seed_str}")
    print(f"Bio mask density: {results['meta']['bio_mask_density']:.4f}")
    print(f"Connectome:       {results['meta']['connectome']}")
    print("="*80)


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "WiringMatters Stage 1 Benchmark — "
            "four-condition experiment (three conditions for sequential tasks)"
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        default="digits",
        help=(
            "Task to run: digits, housing, moons, seq_digits, sequential_mnist, "
            "or 'all' (runs digits+housing+moons+seq_digits, no download). "
            "(default: digits)"
        ),
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=448,
        help=(
            "Hidden layer size (default: 448). "
            "448 matches the C. elegans connectome exactly — the full adjacency "
            "matrix is used as-is, no cropping or tiling. Using a different size "
            "triggers the scaling strategy documented in CLAUDE.md."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help=(
            "Number of seeds to run (default: 1, i.e. seed 42 only). "
            "With --seeds 5 the experiment runs seeds 42, 43, 44, 45, 46 "
            "and reports mean ± std across seeds."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results JSON (default: results/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.task == "all":
        tasks = ALL_DEFAULT_TASKS  # digits, housing, moons, seq_digits
    else:
        tasks = [args.task]

    seeds = list(range(BASE_SEED, BASE_SEED + args.seeds))

    run_benchmark(
        tasks=tasks,
        hidden_size=args.hidden,
        epochs=args.epochs,
        output_dir=Path(args.output),
        seeds=seeds,
    )
