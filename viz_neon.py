"""Dark/neon 3D loss-surface visuals: floor contours + glowing optimizer paths + orbiting camera GIF."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from nn import unpack, loss_fn
from optim import OPTIMIZERS
from compare import train_opt, splits, shared_plane, landscape_grid

NEON = {"SGD": "#ff3b6b", "Momentum": "#ffb703", "Adam": "#00f5d4"}
BG = "#05060a"


def build(epochs=2000, grid=70):
    X, y, _, _ = splits(0)
    paths = {n: train_opt(X, y, mk(), epochs, seed=0)[2] for n, mk in OPTIMIZERS.items()}
    center, d1, d2 = shared_plane(paths)
    M = np.stack([d1, d2]).T
    coords = {n: (p - center) @ M for n, p in paths.items()}
    a, b, Z = landscape_grid(center, d1, d2, np.vstack(list(coords.values())), X, y, grid=grid)
    ploss = {n: np.array([loss_fn(unpack(center + c[0] * d1 + c[1] * d2), X, y) for c in cs]) for n, cs in coords.items()}
    return a, b, Z, coords, ploss


def draw(ax, a, b, Z, coords, ploss, upto=None, zcap=None):
    A, B = np.meshgrid(a, b)
    Z = np.log10(Z)
    ploss = {n: np.log10(v) for n, v in ploss.items()}
    zcap = zcap or Z.max()
    Zc = np.minimum(Z, zcap)
    zmin = Zc.min() - 0.25 * (zcap - Zc.min())
    ax.set_facecolor(BG)
    ax.plot_surface(A, B, Zc, cmap="plasma", alpha=0.78, linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.contour(A, B, Zc, levels=18, zdir="z", offset=zmin, cmap="plasma", linewidths=0.8)
    for n, c in coords.items():
        j = len(c) if upto is None else upto
        z = np.minimum(ploss[n][:j], zcap) + 0.07 * (zcap - zmin)
        ax.plot(c[:j, 0], c[:j, 1], z, color=NEON[n], lw=5, alpha=0.25)   # glow
        ax.plot(c[:j, 0], c[:j, 1], z, color=NEON[n], lw=2, label=n)
        ax.scatter(c[j - 1, 0], c[j - 1, 1], z[-1], color="white", edgecolors=NEON[n], s=60, depthshade=False)
    ax.set_zlim(zmin, zcap)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0)); axis._axinfo["grid"]["color"] = (1, 1, 1, 0.08)
        axis.label.set_color("#9aa0b4"); axis.set_tick_params(colors="#5b6075", labelsize=7)
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2"); ax.set_zlabel("log10 loss")
    leg = ax.legend(loc="upper left", facecolor=BG, edgecolor="#333", labelcolor="white", fontsize=8)


def run(out="results", frames=72):
    a, b, Z, coords, ploss = build()
    fig = plt.figure(figsize=(8, 6.5), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    draw(ax, a, b, Z, coords, ploss)
    ax.view_init(elev=58, azim=-105)
    ax.set_title("Loss landscape: SGD vs Momentum vs Adam", color="white", fontsize=12)
    fig.savefig(f"{out}/neon_landscape.png", dpi=170, facecolor=BG)
    plt.close(fig)

    # orbiting-camera GIF while paths get drawn in the first half
    fig = plt.figure(figsize=(6, 5), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    n_pts = len(next(iter(coords.values())))
    reveal = np.unique(np.geomspace(2, n_pts, frames // 2).astype(int))

    def upd(k):
        ax.clear()
        upto = reveal[min(k, len(reveal) - 1)]
        draw(ax, a, b, Z, coords, ploss, upto=upto)
        ax.view_init(elev=58, azim=-105 + k * (360 / frames))
        ax.set_title("Gradient descent on the loss surface", color="white", fontsize=10)

    FuncAnimation(fig, upd, frames=frames).save(f"{out}/neon_orbit.gif", writer=PillowWriter(fps=15), dpi=80,
                                                savefig_kwargs={"facecolor": BG})
    plt.close(fig)


if __name__ == "__main__":
    run()
