"""
STEP 4 - COMPREHENSIVE MODEL COMPARISON & VISUALIZATION
Compares TextCNN Baseline vs. CNN-BiLSTM-Attention across SST-2 and YOSM

Evaluates and summarizes:
  - Test Accuracy, Precision, Recall, F1 Score
  - Trainable Parameter Counts
  - Training Time & Convergence Epoch
  - Confusion Matrix Visualizations
  - Outputs structured JSON, CSV, and comparative multi-panel plots
"""

import os
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
DATASETS    = ["sst2", "yosm"]
MODELS      = [
    ("TextCNN", "cnn_baseline_metrics.json"),
    ("CNN-BiLSTM-Attention", "cnn_bilstm_metrics.json"),
]


def load_metrics():
    """Load metrics from JSON artifacts for all model and dataset combinations."""
    records = []
    for ds in DATASETS:
        ds_name = "SST-2 (English)" if ds == "sst2" else "YOSM (Yoruba)"
        for model_name, filename in MODELS:
            path = os.path.join(RESULTS_DIR, ds, filename)
            if not os.path.exists(path):
                print(f"[warn] Missing metrics file: {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            t = data.get("test", {})
            cm = t.get("confusion_matrix", [[0, 0], [0, 0]])
            records.append({
                "dataset_code": ds,
                "dataset": ds_name,
                "model": model_name,
                "params": data.get("params", 0),
                "best_epoch": data.get("best_epoch", 0),
                "train_time_sec": data.get("train_time_sec", 0.0),
                "accuracy": t.get("accuracy", 0.0),
                "precision": t.get("precision", 0.0),
                "recall": t.get("recall", 0.0),
                "f1": t.get("f1", 0.0),
                "confusion_matrix": cm,
            })
    return records


def export_csv_and_json(records):
    """Export summary tables to CSV and JSON."""
    json_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Comparison JSON saved -> {json_path}")

    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    fieldnames = [
        "dataset", "model", "accuracy", "precision", "recall", "f1",
        "params", "best_epoch", "train_time_sec"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Comparison CSV saved  -> {csv_path}")


def export_markdown_table(records):
    """Generate and return a formatted GitHub Markdown table."""
    md = []
    md.append("| Dataset | Model | Accuracy | Precision | Recall | F1 Score | Parameters | Best Epoch | Train Time |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in records:
        md.append(
            f"| **{r['dataset']}** | {r['model']} "
            f"| {r['accuracy']*100:.2f}% "
            f"| {r['precision']*100:.2f}% "
            f"| {r['recall']*100:.2f}% "
            f"| **{r['f1']*100:.2f}%** "
            f"| {r['params']:,} "
            f"| {r['best_epoch']} "
            f"| {r['train_time_sec']:.1f}s |"
        )
    table_str = "\n".join(md)

    md_path = os.path.join(RESULTS_DIR, "model_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison Summary\n\n" + table_str + "\n")
    print(f"Comparison Markdown saved -> {md_path}")
    return table_str


def generate_comparison_plots(records):
    """Generate comparative visualization plots across models and datasets."""
    if not records:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    datasets = ["SST-2 (English)", "YOSM (Yoruba)"]
    models = ["TextCNN", "CNN-BiLSTM-Attention"]
    colors = ["#1f77b4", "#2ca02c"]

    # 1. Accuracy comparison
    x = np.arange(len(datasets))
    width = 0.35
    for idx, model_name in enumerate(models):
        accs = []
        for ds in datasets:
            m = next((r for r in records if r["dataset"] == ds and r["model"] == model_name), None)
            accs.append(m["accuracy"] * 100 if m else 0.0)
        axes[0].bar(x + (idx - 0.5) * width, accs, width, label=model_name, color=colors[idx])
        for i, val in enumerate(accs):
            if val > 0:
                axes[0].text(x[i] + (idx - 0.5) * width, val + 1, f"{val:.1f}%",
                             ha="center", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Test Accuracy Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

    # 2. F1 Score comparison
    for idx, model_name in enumerate(models):
        f1s = []
        for ds in datasets:
            m = next((r for r in records if r["dataset"] == ds and r["model"] == model_name), None)
            f1s.append(m["f1"] * 100 if m else 0.0)
        axes[1].bar(x + (idx - 0.5) * width, f1s, width, label=model_name, color=colors[idx])
        for i, val in enumerate(f1s):
            if val > 0:
                axes[1].text(x[i] + (idx - 0.5) * width, val + 1, f"{val:.1f}%",
                             ha="center", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("F1 Score (%)")
    axes[1].set_title("Test F1 Score Comparison")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")

    # 3. Model Parameters comparison
    for idx, model_name in enumerate(models):
        param_counts = []
        for ds in datasets:
            m = next((r for r in records if r["dataset"] == ds and r["model"] == model_name), None)
            param_counts.append(m["params"] / 1e6 if m else 0.0)
        axes[2].bar(x + (idx - 0.5) * width, param_counts, width, label=model_name, color=colors[idx])
        for i, val in enumerate(param_counts):
            if val > 0:
                axes[2].text(x[i] + (idx - 0.5) * width, val + 0.05, f"{val:.2f}M",
                             ha="center", fontsize=9, fontweight="bold")
    axes[2].set_ylabel("Parameters (Millions)")
    axes[2].set_title("Model Complexity (Parameters)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(datasets)
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "model_comparison_plots.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Comparison plots saved    -> {plot_path}")


def main():
    print("--- Generating Model Comparison Report ---")
    records = load_metrics()
    if not records:
        print("No metrics records found to compare.")
        return

    export_csv_and_json(records)
    table_str = export_markdown_table(records)
    generate_comparison_plots(records)

    print("\n" + table_str)


if __name__ == "__main__":
    main()
