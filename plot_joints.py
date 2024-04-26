import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D


data = np.load('./amass-master/support_data/github_data/amass_sample.npz', allow_pickle=True)
marker_data = data['marker_data'][0]
labels = data['marker_labels']
print(labels)

def skeleton_plot(markerData):
    # Unpack the list of marker data into x, y, z coordinates
    markerX, markerY, markerZ = zip(*markerData)
    
    # Create a new figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot lines between each adjacent vertex using a 3D line plot
    ax.plot(markerX, markerY, markerZ, marker='o', linestyle='-', color='blue')  # 'o' creates a circle at each vertex
    
    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

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

scatter3d(marker_data[:,0], marker_data[:,1], marker_data[:,2], np.arange(len(marker_data)), labels=labels)

skeleton_plot(marker_data)