"""
STEP 2 - TEXTCNN BASELINE PIPELINE
Sentiment classification baseline for SST-2 (English) and YOSM (Yoruba)

Architecture:
    Embedding (trainable) -> Parallel 1D Convolutions (kernels 2, 3, 4, 5)
                          -> ReLU -> Global Max Pooling per kernel branch
                          -> Concatenated features (dim 400)
                          -> Dropout (p=0.5) -> Dense Linear (1 output logit)
                          -> BCEWithLogitsLoss / Sigmoid classification

Outputs:
    - Checkpoint: results/{dataset}/cnn_baseline.pt
    - Metrics JSON: results/{dataset}/cnn_baseline_metrics.json
    - Training curves: results/{dataset}/cnn_training_curves.png
    - Confusion matrix: results/{dataset}/cnn_confusion_matrix.png
"""

import os
import json
import time
import pickle
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

SEED         = 42
DATA_ROOT    = "data/processed"
RESULTS_ROOT = "results"
DATASETS     = ("yosm", "sst2")

EMB_DIM      = 200
FILTERS      = 100            # conv filters per kernel size
KERNELS      = (2, 3, 4, 5)   # 2- to 5-gram feature detectors
DROPOUT      = 0.5
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 64
EPOCHS       = 20
PATIENCE     = 5              # early stopping patience
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = SEED):
    """Seed all random number generators for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextCNN(nn.Module):
    """1D Multi-Kernel TextCNN Sentiment Classifier (Kim, 2014 style)."""
    def __init__(self, vocab_size, emb_dim=EMB_DIM, kernels=KERNELS,
                 filters=FILTERS, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, filters, kernel_size=k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernels), 1)

    def forward(self, x):
        # x: (B, L) token IDs
        e = self.embedding(x).transpose(1, 2)  # (B, emb_dim, L)
        # Apply 1D conv + ReLU + 1D Global Max-Pool over temporal dimension
        feats = [torch.relu(conv(e)).max(dim=2).values for conv in self.convs]
        h = self.dropout(torch.cat(feats, dim=1))  # (B, filters * len(kernels))
        return self.fc(h).squeeze(-1)              # (B,) raw logits


def load_split(name, data_dir):
    """Load preprocessed split (.npz) containing x (int32) and y (int64/float32)."""
    d = np.load(os.path.join(data_dir, f"{name}.npz"))
    return d["x"], d["y"].astype(np.float32)


def make_loader(x, y, batch_size=BATCH_SIZE, shuffle=False):
    """Construct PyTorch DataLoader from numpy arrays."""
    tx = torch.from_numpy(x).long()
    ty = torch.from_numpy(y)
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def run_eval(model, loader, loss_fn=None):
    """Evaluate model on a DataLoader, returning classification metrics and loss."""
    model.eval()
    ys, preds, losses, n = [], [], [], 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        if loss_fn is not None:
            losses.append(loss_fn(logits, yb).item() * len(xb))
            n += len(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        ys.extend(yb.cpu().numpy().astype(int))
        preds.extend((probs >= 0.5).astype(int))

    y_true, y_pred = np.asarray(ys), np.asarray(preds)
    out = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if loss_fn is not None:
        out["loss"] = float(sum(losses) / max(n, 1))
    return out, y_true, y_pred


def train_model(model, train_loader, val_loader, y_train, epochs=EPOCHS,
                patience=PATIENCE, lr=LR, weight_decay=WEIGHT_DECAY,
                monitor="f1"):
    """Train TextCNN with AdamW, BCEWithLogitsLoss, and validation early stopping."""
    neg, pos = float((y_train == 0).sum()), float((y_train == 1).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_score = 0.0 if monitor == "f1" else float("inf")
    best_state = None
    best_epoch = 0
    wait = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * len(xb)
            seen += len(xb)

        val, _, _ = run_eval(model, val_loader, loss_fn)
        train_loss = running / max(seen, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["val_f1"].append(val["f1"])

        current_score = val["f1"] if monitor == "f1" else val["loss"]
        improved = (current_score > best_score) if monitor == "f1" else (current_score < best_score)

        marker = ""
        if improved:
            best_score = current_score
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        else:
            wait += 1

        print(f"epoch {epoch:02d} | train_loss {train_loss:.4f} "
              f"| val_loss {val['loss']:.4f} | val_acc {val['accuracy']:.4f} "
              f"| val_f1 {val['f1']:.4f}{marker}")

        if wait >= patience:
            print(f"Early stopping triggered (no improvement for {patience} epochs).")
            break

    total_time = time.time() - t0
    return history, total_time, best_state, best_epoch


def plot_history(history, results_dir, prefix="cnn"):
    """Plot and save training/validation loss and accuracy/F1 trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history["train_loss"], label="train", color="#1f77b4")
    axes[0].plot(history["val_loss"], label="val", color="#ff7f0e")
    axes[0].set_title("Loss Trajectory")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(history["val_acc"], label="val acc", color="#2ca02c")
    axes[1].plot(history["val_f1"], label="val F1", color="#d62728")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{prefix}_training_curves.png"), dpi=150)
    plt.close(fig)


def plot_cm(cm, results_dir, prefix="cnn"):
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues", interpolation="nearest")
    thresh = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative (0)", "Positive (1)"])
    ax.set_yticklabels(["Negative (0)", "Positive (1)"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix ({prefix.upper()})")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{prefix}_confusion_matrix.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="TextCNN sentiment baseline")
    parser.add_argument("--dataset", choices=DATASETS, default="yosm",
                        help="Preprocessed dataset to train/evaluate on (yosm or sst2)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience")
    parser.add_argument("--monitor", choices=["f1", "loss"], default="f1",
                        help="Metric to monitor for early stopping")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate existing saved checkpoint without retraining")
    args = parser.parse_args()

    data_dir    = os.path.join(DATA_ROOT, args.dataset)
    results_dir = os.path.join(RESULTS_ROOT, args.dataset)

    seed_everything(SEED)
    os.makedirs(results_dir, exist_ok=True)

    x_tr, y_tr = load_split("train", data_dir)
    x_va, y_va = load_split("val", data_dir)
    x_te, y_te = load_split("test", data_dir)
    print(f"[{args.dataset.upper()}] train: {len(x_tr)} | val: {len(x_va)} | test: {len(x_te)} | device: {DEVICE}")

    vocab_file = os.path.join(data_dir, "vocab.pkl")
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    vocab_size = max(vocab.values()) + 1

    train_loader = make_loader(x_tr, y_tr, batch_size=args.batch_size, shuffle=True)
    val_loader   = make_loader(x_va, y_va, batch_size=args.batch_size, shuffle=False)
    test_loader  = make_loader(x_te, y_te, batch_size=args.batch_size, shuffle=False)

    model = TextCNN(vocab_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TextCNN | vocab: {vocab_size} | trainable parameters: {n_params:,}\n")

    ckpt_path = os.path.join(results_dir, "cnn_baseline.pt")
    metrics_path = os.path.join(results_dir, "cnn_baseline_metrics.json")

    if args.eval_only:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        print(f"Loading existing checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        test, y_true, y_pred = run_eval(model, test_loader)
        print("\n===== TEST RESULTS (EVAL ONLY) =====")
        print(classification_report(y_true, y_pred, target_names=["negative", "positive"], zero_division=0))
        print(f"Accuracy:  {test['accuracy']:.4f}")
        print(f"Precision: {test['precision']:.4f}")
        print(f"Recall:    {test['recall']:.4f}")
        print(f"F1 Score:  {test['f1']:.4f}")
        print(f"Confusion Matrix:\n{np.asarray(test['confusion_matrix'])}")
        return

    history, train_time, best_state, best_epoch = train_model(
        model, train_loader, val_loader, y_tr, epochs=args.epochs,
        patience=args.patience, lr=args.lr, monitor=args.monitor
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    test, y_true, y_pred = run_eval(model, test_loader)

    print(f"\n===== TEST RESULTS - TEXTCNN ({args.dataset.upper()}) =====")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"], zero_division=0))
    print(f"Confusion Matrix:\n{np.asarray(test['confusion_matrix'])}")
    print(f"Accuracy:  {test['accuracy']:.4f}")
    print(f"Precision: {test['precision']:.4f}")
    print(f"Recall:    {test['recall']:.4f}")
    print(f"F1 Score:  {test['f1']:.4f}")
    print(f"Best Epoch: {best_epoch} | Train Time: {train_time:.1f}s")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "TextCNN",
            "dataset": args.dataset,
            "params": n_params,
            "best_epoch": best_epoch,
            "train_time_sec": round(train_time, 2),
            "test": test,
            "history": history
        }, f, indent=2)

    torch.save({
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "vocab_size": vocab_size,
        "config": {"emb_dim": EMB_DIM, "kernels": list(KERNELS), "filters": FILTERS}
    }, ckpt_path)

    plot_history(history, results_dir, prefix="cnn")
    plot_cm(np.asarray(test["confusion_matrix"]), results_dir, prefix="cnn")
    print(f"\nArtifacts successfully saved to {results_dir}/")


if __name__ == "__main__":
    main()