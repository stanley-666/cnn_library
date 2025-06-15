import numpy as np
import matplotlib.pyplot as plt

W, H = 128, 128  # 根據你實際大小

# 讀入 .raw 檔（float32 binary）
arr = np.fromfile("heapmap/density_map.raw", dtype=np.float32).reshape((H, W))

# 儲存成 numpy .npy 格式
np.save("heapmap/density_map.npy", arr)

# 載入儲存好的 npy
arr = np.load("heapmap/density_map.npy")
# 顯示
plt.imshow(arr, cmap="jet")
plt.colorbar()
plt.title("Heatmap from PGM")
plt.show()
