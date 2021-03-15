import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

disp_x = np.genfromtxt('../Star_frames/Noiseless_frames/flow/Star_disp_x.csv', delimiter=',')
disp_y = np.genfromtxt('../Star_frames/Noiseless_frames/flow/Star_disp_y.csv', delimiter=',')
fig, ax = plt.subplots(figsize=(disp_x.shape[1]/200, disp_x.shape[0]/200))
c = plt.pcolormesh(disp_x + disp_y,vmin = -0.5, vmax = 0.5)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(c, cax=cax)
plt.show()