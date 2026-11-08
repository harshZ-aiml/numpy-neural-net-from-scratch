"""Run the whole project: gradient check -> train -> plots -> loss landscape -> GIF."""
import os
import numpy as np
from nn import make_data, train, accuracy
from gradient_check import run_check
from plots import plot_loss, plot_boundary
from landscape import plot_landscape, make_gif
from compare import run as run_compare
from viz_neon import run as run_neon

OUT = "results"
os.makedirs(OUT, exist_ok=True)

print("1) Gradient check")
run_check()

print("\n2) Training")
X, y = make_data()
params, losses, path = train(X, y)
print(f"final loss {losses[-1]:.4f} | accuracy {accuracy(params, X, y):.1%}")
np.save(f"{OUT}/path.npy", path)  # weight trajectory

print("\n3) Plots")
plot_loss(losses, OUT)
plot_boundary(params, X, y, OUT)

print("4) Loss landscape + descent path")
a, b, Z, coords = plot_landscape(path, X, y, OUT)

print("5) GIF")
make_gif(a, b, Z, coords, OUT)

print("\n6) Optimizer comparison (SGD / Momentum / Adam, 10 seeds, held-out test set)")
run_compare(OUT)

print("\n7) Neon 3D visuals")
run_neon(OUT)

print(f"\nDone. See the '{OUT}/' folder.")
