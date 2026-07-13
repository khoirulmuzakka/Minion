import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    x = np.linspace(-100.0, 100.0, 4001)
    u = x**2

    transforms = [
        (r"Linear: $\phi(u)=u$", u),
        (r"Power: $\phi(u)=u^2/10^4$", u**2 / 1.0e4),
        (r"Log: $\phi(u)=2000\log(1+0.01u)$", 2000.0 * np.log1p(0.01 * u)),
        (r"Cosine: $\phi(u)=1-\cos(0.01u)$", 1.0 - np.cos(0.01 * u)),
        (r"Repeated optima: $\phi(u)=u(u-2500)^2/6.25{\cdot}10^7$", u * (u - 2500.0) ** 2 / 6.25e7),
        (r"Oscillatory: $\phi(u)=u(1+0.45\sin(0.02u))$", u * (1.0 + 0.45 * np.sin(0.02 * u))),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), constrained_layout=True)

    for ax, (title, y) in zip(axes.ravel(), transforms):
        ax.plot(x, y, color="#1f4e79", linewidth=1.5)
        ax.axvline(0.0, color="#666666", linewidth=0.7, linestyle="--", alpha=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-100.0, 100.0)
        ax.set_xlabel(r"$x$", fontsize=9)
        ax.set_ylabel(r"$\phi(g(x))$", fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.tick_params(labelsize=8)

    fig.suptitle(r"Value transformations applied to $g(x)=x^2$ on $[-100,100]$", fontsize=12)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "value_transform_quadratic.pdf"))
    fig.savefig(os.path.join(out_dir, "value_transform_quadratic.png"), dpi=200)


if __name__ == "__main__":
    main()
