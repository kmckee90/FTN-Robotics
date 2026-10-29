"""Headless training result plots."""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_matrix(matrix, names, out_path, phases_done):
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(8, .42*n+5), max(6, .42*n+3)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap)
    labels = [f"{i+1}: {name.removesuffix('-v3')}" for i, name in enumerate(names)]
    ax.set_xticks(range(n), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(n + 1), ["0: prior", *labels], fontsize=8)
    ax.axhline(.5, color="white", linewidth=.7)
    ax.plot([-.5, n-.5], [.5, n+.5], color="white", linewidth=.7)
    ax.set_xlabel("Evaluation task (upper triangle: transfer)")
    ax.set_ylabel("Trained through task (lower triangle: backward recall)")
    ax.set_title(f"Sequential MT50 BC success rate — {phases_done}/{n} tasks complete")
    for i, j in zip(*np.where(np.isfinite(matrix))):
        v = matrix[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5 if n > 20 else 8,
                color="white" if v < .5 else "black")
    fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    fig.tight_layout()
    tmp = Path(out_path).with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=150)
    plt.close(fig)
    os.replace(tmp, out_path)

def plot_losses(losses, out_path):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set(xlabel="Epoch", ylabel="Mean final batch BC loss", title="MT10 parallel segments")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
