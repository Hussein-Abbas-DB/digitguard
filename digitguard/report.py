"""Render a reproducible benchmark graphic from the training report."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def render(directory):
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    fig, (matrix, bars) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained",
                                      gridspec_kw={"width_ratios": [1, 1.5]})
    fig.suptitle("DigitGuard | Held-out evaluation", fontsize=16, fontweight="bold")
    matrix.set_title("Clean confusion matrix")
    counts = np.array(metrics["confusion_matrix"])
    matrix.imshow(counts, cmap="Blues")
    for row, col in zip(*np.nonzero(counts)):
        count = counts[row, col]
        matrix.text(col, row, str(count), ha="center", va="center",
                    color="white" if count > 10 else "#1e293b")
    matrix.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit")

    cases = [metrics["test"], *metrics["stress"].values()]
    positions = np.arange(len(cases))
    for offset, key, label, color in [
        (-0.18, "accuracy", "Forced prediction accuracy", "#285a87"),
        (0.18, "acceptance_rate", "Acceptance rate", "#1c876e"),
    ]:
        values = [100 * case[key] for case in cases]
        containers = bars.barh(positions + offset, values, height=0.32, label=label, color=color)
        bars.bar_label(containers, fmt="%.1f%%", padding=4, fontsize=9)
    bars.set(yticks=positions, yticklabels=["Clean", "Gaussian noise", "Shift right", "Center occlusion"],
             xlim=(0, 113), xlabel="Percent of 180 test examples per condition")
    bars.invert_yaxis()
    bars.legend(loc="lower right", fontsize=9)
    bars.grid(axis="x", alpha=0.2)
    bars.set_axisbelow(True)
    fig.savefig(directory / "benchmark.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    render(parser.parse_args().report_dir)
