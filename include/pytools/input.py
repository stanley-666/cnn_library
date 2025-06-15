from PIL import Image
import numpy as np

img = Image.open("41.jpg").convert("RGB")
print("Image size:", img.size)  # (W, H)

# 歸一化並映射到 [-1,1]
img_array = np.array(img).astype(np.float32) # (H,W) NORM 0~255 / 255
img_array = img_array - 127

# 改成 channel-first
img_ch_first = np.transpose(img_array, (2, 0, 1))  # (C, H, W)

# Flatten 並輸出成 C 陣列
flat_array = img_ch_first.flatten()
c_array_name = "image_data"
c_type = "static const float"
c_elements = ", ".join([f"{v:.6f}f" for v in flat_array])
c_array_code = f"{c_type} {c_array_name}[{len(flat_array)}] = {{{c_elements}}};"

with open("image_data_array.h", "w") as f:
    f.write(c_array_code)

print("✅ 已輸出 C 陣列，共", len(flat_array), "個元素。")
