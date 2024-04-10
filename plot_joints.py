import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

# Load data
path1 = './amass-master/support_data/github_data/amass_sample.npz'
data = np.load(path1, allow_pickle=True)
marker_data = data['marker_data']
print(list(data.keys()))
# It's redundant to load the data twice and print the keys twice, so these lines are removed.
labels = data['marker_labels']

def scatter3d(x, y, z, cs, colorsMap='jet', labels=None):
    cm = plt.get_cmap(colorsMap)
    cNorm = mcolors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
    fig = plt.figure(figsize=(10, 7))  # Adjusted figure size for better visualization
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    
    # Annotate each point in 3D space with its label
    if labels is not None:
        for i, txt in enumerate(labels):
            ax.text(x[i], y[i], z[i], txt, size=5)
    
    # Create a colorbar
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # Adjust these values as needed to prevent overlap
    fig.colorbar(scalarMap, cax=cbar_ax, aspect=5)  # Removed shrink parameter for clarity

    plt.show()

# Adjusting the color scale input to match the length of marker_data instead of data
scatter3d(marker_data[:, 0], marker_data[:, 1], marker_data[:, 2], np.arange(marker_data.shape[0]), labels=labels)