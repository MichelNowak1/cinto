
import json
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl

imap_file = open("finite_differences.json",)
imap = json.load(imap_file)

x = np.array(imap.get("x_bounds"))
y = np.array(imap.get("y_bounds"))
z = np.array(imap.get("z_bounds"))
e = imap.get("energy_bounds")
importance = np.array(imap.get("importance"))

num_cells = (len(x)-1)*(len(y)-1)*(len(z)-1)*(len(e)-1)
importance=importance.reshape((len(x)-1,len(y)-1,len(z)-1))
importance[importance==np.nan] = 0
importance[importance==np.inf] = 0
importance[importance==None] = 0

xx = []
yy = []
strength = []

y, x = np.mgrid[slice(-40, 40, 1),
                slice(-40, 40, 1)]

for i in range(importance.shape[0]):
    for j in range(importance.shape[1]):
        strength.append(importance[i, j])

strength=np.array(strength).reshape((importance.shape[0], importance.shape[1]))

fig, ax = plt.subplots(figsize=(7,7))
Q = ax.contourf(x, y, strength, cmap="viridis", linewidths=(3,), extent=[-40,40,-40,40],norm=mpl.colors.LogNorm())
ax.set_aspect('equal')
ax.set_title("scalar adjoint flux for bypass")
ax.set_facecolor('black')
fig.colorbar(Q, shrink=0.8)
plt.savefig("scalar.png", bbox_inches='tight', dpi=200)
