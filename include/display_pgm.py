import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("density_map.pgm")
plt.imshow(img, cmap="jet")  # 或 cmap="gray"
plt.colorbar()
plt.show()
