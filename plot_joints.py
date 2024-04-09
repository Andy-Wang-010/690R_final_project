import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

# Load data
data = np.load('./amass-master/support_data/github_data/amass_sample.npz', allow_pickle=True)
a = data['marker_data'][0]
data_trans = data['trans']
labels = data['marker_labels']
normalized = a-data_trans

def scatter3d(x, y, z, cs, colorsMap='jet', labels=None):
    cm = plt.get_cmap(colorsMap)
    cNorm = mcolors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    
    # Annotate each point in 3D space with its label
    if labels is not None:
        for i, txt in enumerate(labels):
            ax.text(x[i], y[i], z[i], txt, size=5)

    # Create a colorbar
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scalarMap, cax=cbar_ax, shrink=0.5, aspect=5)

    plt.show()

# Call the function with marker labels
scatter3d(normalized[:,0], normalized[:,1], normalized[:,2], np.arange(len(normalized)), labels=labels)