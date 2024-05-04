import sys
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from plot_joints import scatter3d


data = np.genfromtxt('data/AllCoords/Male1_B26_WalkToSkip.csv', skip_header=1, delimiter=',')

fig, scatter = scatter3d(data[0,1::3],data[0,3::3],data[0,3::3],np.ones(6))

def update(idx, data, scatter):
    scatter._offsets3d = (data[idx,1::3],data[idx,3::3],data[idx,2::3])

ani = anim.FuncAnimation(fig, update, len(data), fargs=(data,scatter))
plt.show()