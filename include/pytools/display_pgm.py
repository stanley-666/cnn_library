import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


img = plt.imread("heapmap/density_map.pgm")
data = np.array(img) # new

plt.imshow(img, cmap="jet")  # 或 cmap="gray"
plt.colorbar()
plt.title("Heatmap from PGM")
plt.show()
