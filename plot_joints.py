import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

data = np.load('./amass-master/support_data/github_data/amass_sample.npz', allow_pickle=True)
a = data['marker_data'][0]
labels = data['marker_labels']
print(labels)

def scatter3d(x, y, z, cs, colorsMap='jet', labels=None):
    cm = plt.get_cmap(colorsMap)
    cNorm = mcolors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if labels is not None:
        for i, txt in enumerate(labels):
            ax.text(x[i], y[i], z[i], txt, size=5)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scalarMap, cax=cbar_ax, shrink=0.5, aspect=5)
    plt.show()

scatter3d(a[:,0], a[:,1], a[:,2], np.arange(len(a)), labels=labels)