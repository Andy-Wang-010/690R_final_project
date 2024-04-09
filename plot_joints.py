import numpy as np
import matplotlib.pyplot as plt 
import matplotlib

a = dict(np.load('./amass/support_data/github_data/amass_sample.npz'))
a = a['marker_data'][0]

import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    # fig.colorbar(scalarMap)
    plt.show()

scatter3d(a[:,0],a[:,1],a[:,2],np.arange(len(a)))