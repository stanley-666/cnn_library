import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
from matplotlib import pyplot as plt
from matplotlib import cm
def load_density_map(txt_file_path, shape=(128, 128)):
    """讀取txt檔案中的密度圖數據並進行 reshape"""
    density_map = np.loadtxt(txt_file_path)
    density_map = density_map.reshape(shape)
    plt.imshow(density_map, cmap=cm.jet)
    plt.show()
    return density_map


txt_file_path = '../Deploy/People_Weight/heatmap.txt'  # 替換為你的txt檔案路徑
density_map = load_density_map(txt_file_path)