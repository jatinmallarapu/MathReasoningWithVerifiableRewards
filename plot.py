"""Plot GRPO reward and sampled completion length from TRL log history."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main(history: Path, output: Path):
    rows = json.loads(history.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for ax, names, label in zip(
        axes, [("reward",), ("completions/mean_length", "completion_length")],
        ["Mean reward", "Mean generated tokens"], strict=True,
    ):
        points = [
            (row["step"], row[name]) for row in rows for name in names if name in row
        ]
        if not points:
            raise ValueError(f"Missing logged metric: {label}")
        ax.plot(*zip(*points), marker=".", linewidth=1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Optimizer step")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.history, args.output)
