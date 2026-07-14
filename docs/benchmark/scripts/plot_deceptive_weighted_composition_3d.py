import os

import matplotlib.pyplot as plt
import numpy as np


def rastrigin(x, y):
    return 20.0 + x**2 + y**2 - 10.0 * (np.cos(2.0 * np.pi * x) + np.cos(2.0 * np.pi * y))


def ackley(x, y):
    r = np.sqrt(0.5 * (x**2 + y**2))
    c = 0.5 * (np.cos(2.0 * np.pi * x) + np.cos(2.0 * np.pi * y))
    return -20.0 * np.exp(-0.2 * r) - np.exp(c) + np.e + 20.0


def griewank(x, y):
    return 1.0 + (x**2 + y**2) / 4000.0 - np.cos(x) * np.cos(y / np.sqrt(2.0))


def steep_ellipsoid(x, y):
    return x**2 + 70.0 * y**2


def rosenbrock_at_zero(x, y):
    u = x + 1.0
    v = y + 1.0
    return 100.0 * (v - u**2) ** 2 + (1.0 - u) ** 2


def sphere(x, y):
    return x**2 + y**2


def normalize(values):
    return values / np.percentile(values, 95.0)


def deceptive_composition(xx, yy, global_fun, deceptive_fun, global_scale, deceptive_scale, offset):
    x_global = np.array([-45.0, -35.0])
    x_deceptive = np.array([45.0, 35.0])

    gx = (xx - x_global[0]) / global_scale
    gy = (yy - x_global[1]) / global_scale
    dx = (xx - x_deceptive[0]) / deceptive_scale
    dy = (yy - x_deceptive[1]) / deceptive_scale

    q_global = normalize(global_fun(gx, gy))
    q_deceptive = normalize(deceptive_fun(dx, dy)) + offset

    d_global = (xx - x_global[0]) ** 2 + (yy - x_global[1]) ** 2
    d_deceptive = (xx - x_deceptive[0]) ** 2 + (yy - x_deceptive[1]) ** 2
    sigma2 = 42.0**2

    raw_global = np.exp(-d_global / (2.0 * sigma2))
    raw_deceptive = np.exp(-d_deceptive / (2.0 * sigma2))
    w_global = raw_global / (raw_global + raw_deceptive)
    w_deceptive = raw_deceptive / (raw_global + raw_deceptive)

    local_radius2 = 24.0**2
    global_core = d_global <= local_radius2
    deceptive_core = d_deceptive <= local_radius2
    w_global[global_core] = 1.0
    w_deceptive[global_core] = 0.0
    w_global[deceptive_core] = 0.0
    w_deceptive[deceptive_core] = 1.0

    f = w_global * q_global + w_deceptive * q_deceptive
    return f, x_global, x_deceptive, offset


def main():
    x = np.linspace(-100.0, 100.0, 161)
    y = np.linspace(-100.0, 100.0, 161)
    xx, yy = np.meshgrid(x, y)

    examples = [
        ("Rastrigin + steep Ellipsoid", rastrigin, steep_ellipsoid, 12.0, 18.0, 0.16),
        ("Ackley + Rosenbrock", ackley, rosenbrock_at_zero, 13.0, 5.8, 0.14),
        ("Griewank + Sphere", griewank, sphere, 7.0, 20.0, 0.12),
    ]

    panels = []
    for title, global_fun, deceptive_fun, global_scale, deceptive_scale, offset in examples:
        f, x_global, x_deceptive, deceptive_offset = deceptive_composition(
            xx,
            yy,
            global_fun,
            deceptive_fun,
            global_scale,
            deceptive_scale,
            offset,
        )
        panels.append((title, np.log1p(f), x_global, x_deceptive, deceptive_offset))

    vmax = max(np.percentile(zz, 98.8) for _, zz, _, _, _ in panels)

    fig = plt.figure(figsize=(14.5, 5.3), constrained_layout=True)
    for idx, (title, zz, x_global, x_deceptive, deceptive_offset) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.plot_surface(
            xx,
            yy,
            zz,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            rcount=95,
            ccount=95,
            vmin=0.0,
            vmax=vmax,
        )
        ax.contour(xx, yy, zz, zdir="z", offset=0.0, levels=13, cmap="viridis", linewidths=0.4)
        ax.scatter([x_global[0]], [x_global[1]], [0.0], marker="*", color="white", edgecolor="black", s=90)
        ax.scatter(
            [x_deceptive[0]],
            [x_deceptive[1]],
            [np.log1p(deceptive_offset)],
            marker="o",
            color="crimson",
            edgecolor="black",
            s=58,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$x_1$", fontsize=8)
        ax.set_ylabel(r"$x_2$", fontsize=8)
        ax.set_zlabel(r"$\log(1+f)$", fontsize=8)
        ax.set_xlim(-100.0, 100.0)
        ax.set_ylim(-100.0, 100.0)
        ax.set_zlim(0.0, vmax)
        ax.view_init(elev=29, azim=-132)
        ax.tick_params(labelsize=7)

    fig.suptitle("Deceptive weighted compositions with different base-function pairs", fontsize=12)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "deceptive_weighted_composition_3d.pdf"))
    fig.savefig(os.path.join(out_dir, "deceptive_weighted_composition_3d.png"), dpi=200)


if __name__ == "__main__":
    main()
