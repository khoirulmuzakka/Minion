import os

import matplotlib.pyplot as plt
import numpy as np


def shifted_sphere(x, y):
    return x**2 + y**2


def shifted_ellipsoid(x, y):
    return x**2 + 25.0 * y**2


def main():
    x = np.linspace(-100.0, 100.0, 401)
    y = np.linspace(-100.0, 100.0, 401)
    xx, yy = np.meshgrid(x, y)

    g1 = shifted_sphere(xx, yy)
    g2 = shifted_ellipsoid(xx, yy)

    z1 = g1 / np.percentile(g1, 95.0)
    z2 = g2 / np.percentile(g2, 95.0)

    w1 = 0.5
    w2 = 0.5
    weighted = w1 * z1 + w2 * z2
    product = (1.0 + z1) * (1.0 + z2) - 1.0
    power_mean = (0.5 * z1**4 + 0.5 * z2**4) ** 0.25

    state_w1 = 0.15 + 0.70 / (1.0 + np.exp(0.08 * xx))
    state_w2 = 1.0 - state_w1
    state_weighted = state_w1 * z1 + state_w2 * z2

    s = z1 + z2
    level_well = s * (1.0 + 0.65 * np.sin(20.0 * s))

    panels = [
        ("Weighted sum", weighted),
        ("Product", product),
        ("Power mean, $p=4$", power_mean),
        ("State-dependent weights", state_weighted),
        ("Nonmonotone level wells", level_well),
    ]
    shown_panels = [(title, np.log1p(values)) for title, values in panels]
    global_max = max(np.percentile(values, 98.5) for _, values in shown_panels)
    levels = np.linspace(0.0, global_max, 32)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (title, shown) in zip(axes_flat, shown_panels):
        contour = ax.contourf(xx, yy, shown, levels=levels, cmap="viridis", extend="max")
        ax.contour(xx, yy, shown, levels=levels[::5], colors="black", linewidths=0.25, alpha=0.35)
        ax.plot(0.0, 0.0, marker="*", color="white", markeredgecolor="black", markersize=11)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$x_1$", fontsize=9)
        ax.set_ylabel(r"$x_2$", fontsize=9)
        ax.set_xlim(-100.0, 100.0)
        ax.set_ylim(-100.0, 100.0)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8)

    axes_flat[-1].axis("off")
    axes_flat[-1].text(
        0.0,
        0.75,
        r"Components:" "\n"
        r"$z_1$: shifted Sphere" "\n"
        r"$z_2$: shifted Ellipsoid" "\n\n"
        r"Color: $\log(1+f)$" "\n"
        r"$\star$: common optimum" "\n\n"
        r"Level wells:" "\n"
        r"$s(1+0.65\sin(20s))$" "\n"
        r"$s=z_1+z_2$",
        fontsize=10,
        va="top",
    )

    fig.colorbar(contour, ax=axes_flat[:5], shrink=0.82, label=r"$\log(1+f)$")
    fig.suptitle("Composition operators applied to shifted Sphere and Ellipsoid components", fontsize=12)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "composition_operators_2d.pdf"))
    fig.savefig(os.path.join(out_dir, "composition_operators_2d.png"), dpi=200)


if __name__ == "__main__":
    main()
