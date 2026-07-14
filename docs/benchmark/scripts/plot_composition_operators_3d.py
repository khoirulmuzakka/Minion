import os

import matplotlib.pyplot as plt
import numpy as np


def sphere(x, y):
    return x**2 + y**2


def ellipsoid(x, y):
    return x**2 + 25.0 * y**2


def main():
    x = np.linspace(-100.0, 100.0, 151)
    y = np.linspace(-100.0, 100.0, 151)
    xx, yy = np.meshgrid(x, y)

    g1 = sphere(xx, yy)
    g2 = ellipsoid(xx, yy)

    z1 = g1 / np.percentile(g1, 95.0)
    z2 = g2 / np.percentile(g2, 95.0)

    weighted = 0.5 * z1 + 0.5 * z2
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
        ("x-dependent weights", state_weighted),
        ("Nonmonotone level wells", level_well),
    ]
    shown_panels = [(title, np.log1p(values)) for title, values in panels]
    global_max = max(np.percentile(values, 98.5) for _, values in shown_panels)

    fig = plt.figure(figsize=(11.0, 7.0), constrained_layout=True)

    for idx, (title, zz) in enumerate(shown_panels, start=1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        ax.plot_surface(
            xx,
            yy,
            zz,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            rcount=90,
            ccount=90,
            vmin=0.0,
            vmax=global_max,
        )
        ax.contour(xx, yy, zz, zdir="z", offset=0.0, levels=12, cmap="viridis", linewidths=0.45)
        ax.scatter([0.0], [0.0], [np.log1p(0.0)], marker="*", color="white", edgecolor="black", s=80)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$x_1$", fontsize=8)
        ax.set_ylabel(r"$x_2$", fontsize=8)
        ax.set_zlabel(r"$\log(1+f)$", fontsize=8)
        ax.set_xlim(-100.0, 100.0)
        ax.set_ylim(-100.0, 100.0)
        ax.set_zlim(0.0, global_max)
        ax.view_init(elev=28, azim=-135)
        ax.tick_params(labelsize=7)

    ax_text = fig.add_subplot(2, 3, 6)
    ax_text.axis("off")
    ax_text.text(
        0.0,
        0.82,
        r"Components:" "\n"
        r"$z_1$: Sphere" "\n"
        r"$z_2$: Ellipsoid" "\n\n"
        r"Height/color: $\log(1+f)$" "\n"
        r"$\star$: common optimum" "\n\n"
        r"Level wells:" "\n"
        r"$s(1+0.65\sin(20s))$" "\n"
        r"$s=z_1+z_2$",
        fontsize=11,
        va="top",
    )

    fig.suptitle("3D views of composition operators applied to Sphere and Ellipsoid components", fontsize=12)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "composition_operators_3d.pdf"))
    fig.savefig(os.path.join(out_dir, "composition_operators_3d.png"), dpi=200)


if __name__ == "__main__":
    main()
