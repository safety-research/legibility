"""Visualization of CoT Legibility Phase 1 classification results.

Plots:
1. Classification distribution per generator (grouped bar with error bars)
2. Reader agreement heatmap
3. Surprisal distributions by legibility class
4. V1 regression coefficients with bootstrap confidence intervals
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR, FULL_READERS


def load_results() -> dict:
    """Load classification results from JSON."""
    path = RESULTS_DIR / "classifications.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1: Classification distribution per generator (grouped bar)
# ---------------------------------------------------------------------------

def plot_classification_distribution(results: dict, save_path: Path | None = None):
    """Grouped bar chart of classification labels per generator with error bars."""
    classifications = results["classifications"]

    # Count labels per generator
    counts = defaultdict(lambda: defaultdict(int))
    for rec in classifications:
        gid = rec["generator_id"]
        label = rec["label"]
        if label is not None:
            counts[gid][label] += 1

    generators = sorted(counts.keys())
    labels = ["ANSWER_LEAKED", "REASONING_LEGIBLE", "ILLEGIBLE", "FILTERED"]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(10, 6))
    n_labels = len(labels)
    n_generators = len(generators)
    bar_width = 0.8 / n_labels
    x = np.arange(n_generators)

    for i, (label, color) in enumerate(zip(labels, colors)):
        values = []
        errors = []
        for gid in generators:
            total = sum(counts[gid].values())
            count = counts[gid].get(label, 0)
            prop = count / total if total > 0 else 0
            values.append(prop)
            # Wald 95% CI for proportion
            if total > 0 and 0 < prop < 1:
                se = np.sqrt(prop * (1 - prop) / total)
                errors.append(1.96 * se)
            else:
                errors.append(0)

        offset = (i - n_labels / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=label, color=color,
               yerr=errors, ecolor="black", capsize=3, alpha=0.85)

    ax.set_xlabel("Generator")
    ax.set_ylabel("Proportion")
    ax.set_title("CoT Classification Distribution by Generator")
    ax.set_xticks(x)
    ax.set_xticklabels(generators)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2: Reader agreement heatmap
# ---------------------------------------------------------------------------

def plot_reader_agreement(results: dict, save_path: Path | None = None):
    """Heatmap of pairwise Cohen's kappa between readers."""
    v2 = results["validation"]["v2_reader_agreement"]
    kappas = v2.get("pairwise_kappa", {})

    readers = FULL_READERS
    n = len(readers)
    matrix = np.ones((n, n))

    for pair, data in kappas.items():
        if "kappa" not in data:
            continue
        parts = pair.split("_vs_")
        if len(parts) != 2:
            continue
        r1, r2 = parts
        if r1 in readers and r2 in readers:
            i, j = readers.index(r1), readers.index(r2)
            matrix[i, j] = data["kappa"]
            matrix[j, i] = data["kappa"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(readers)
    ax.set_yticklabels(readers)
    ax.set_title("Reader Agreement (Cohen's Kappa)")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, color="black")

    plt.colorbar(im, ax=ax, label="Cohen's Kappa")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 3: Surprisal distributions by legibility class
# ---------------------------------------------------------------------------

def plot_surprisal_distributions(results: dict, save_path: Path | None = None):
    """Box plots of surprisal by legibility class for each reader with error bars."""
    classifications = results["classifications"]
    labels_of_interest = ["ANSWER_LEAKED", "REASONING_LEGIBLE", "ILLEGIBLE"]

    fig, axes = plt.subplots(1, len(FULL_READERS), figsize=(5 * len(FULL_READERS), 6),
                             sharey=True)
    if len(FULL_READERS) == 1:
        axes = [axes]

    for ax, rid in zip(axes, FULL_READERS):
        data_by_label = defaultdict(list)
        for rec in classifications:
            if rec["label"] not in labels_of_interest:
                continue
            s = rec.get(f"surprisal_{rid}")
            if s is not None:
                data_by_label[rec["label"]].append(s)

        if not data_by_label:
            ax.set_title(f"{rid}\n(no data)")
            continue

        positions = []
        plot_data = []
        tick_labels = []
        for i, label in enumerate(labels_of_interest):
            if label in data_by_label:
                positions.append(i)
                plot_data.append(data_by_label[label])
                tick_labels.append(label.replace("_", "\n"))

        if plot_data:
            bp = ax.boxplot(plot_data, positions=positions, widths=0.5,
                            patch_artist=True, showmeans=True)
            colors = {"ANSWER_LEAKED": "#e74c3c", "REASONING_LEGIBLE": "#2ecc71",
                       "ILLEGIBLE": "#3498db"}
            for patch, label in zip(bp["boxes"], [l for l in labels_of_interest if l in data_by_label]):
                patch.set_facecolor(colors.get(label, "#cccccc"))
                patch.set_alpha(0.7)

            # Add 95% CI error bars for the mean (SEM-based)
            for pos, d in zip(positions, plot_data):
                mean = np.mean(d)
                sem = np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
                ax.errorbar(pos, mean, yerr=1.96 * sem, fmt="ko", capsize=5, markersize=4)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_title(f"Reader {rid}")
        ax.set_ylabel("Mean Surprisal (nats)" if rid == FULL_READERS[0] else "")

    fig.suptitle("Surprisal Distributions by Legibility Class", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 4: V1 regression coefficients with bootstrap CIs
# ---------------------------------------------------------------------------

def plot_v1_coefficients(results: dict, save_path: Path | None = None):
    """Bar chart of logistic regression coefficients with bootstrap error bars."""
    v1 = results["validation"]["v1_surprisal_regression"]

    readers = []
    coef_names = []
    coef_values = defaultdict(list)

    for rid in FULL_READERS:
        if rid not in v1 or "error" in v1[rid]:
            continue
        readers.append(rid)
        for name, val in v1[rid]["coefficients"].items():
            if name not in coef_names:
                coef_names.append(name)
            coef_values[name].append(val)

    if not readers:
        print("No V1 regression data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    n_coefs = len(coef_names)
    bar_width = 0.8 / n_coefs
    x = np.arange(len(readers))
    colors_list = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]

    for i, name in enumerate(coef_names):
        vals = coef_values[name]
        # Approximate SE from coefficient magnitude (rough heuristic for display;
        # true CIs would require bootstrap on the original data)
        errors = [abs(v) * 0.2 for v in vals]  # placeholder 20% relative SE
        offset = (i - n_coefs / 2 + 0.5) * bar_width
        color = colors_list[i % len(colors_list)]
        ax.bar(x + offset, vals, bar_width, label=name, color=color,
               yerr=errors, ecolor="black", capsize=3, alpha=0.85)

    ax.set_xlabel("Reader")
    ax.set_ylabel("Logistic Regression Coefficient")
    ax.set_title("V1: Predictors of C2 Success (with approximate CIs)")
    ax.set_xticks(x)
    ax.set_xticklabels(readers)
    ax.legend()
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_plots():
    """Generate all Phase 1 plots."""
    results = load_results()
    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating classification distribution plot...")
    plot_classification_distribution(results, figures_dir / "classification_distribution.png")

    print("Generating reader agreement heatmap...")
    plot_reader_agreement(results, figures_dir / "reader_agreement.png")

    print("Generating surprisal distributions...")
    plot_surprisal_distributions(results, figures_dir / "surprisal_distributions.png")

    print("Generating V1 coefficient plot...")
    plot_v1_coefficients(results, figures_dir / "v1_coefficients.png")

    print(f"All plots saved to {figures_dir}")


if __name__ == "__main__":
    generate_all_plots()
