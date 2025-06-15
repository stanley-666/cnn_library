import re
import numpy as np

# === 1. 設定輸入檔案 ===
filename = "Weight1"
input_file = filename +".h"
output_file = "converted_" + filename + ".h"

# === 2. 設定每個陣列的目標形狀 ===
# - 對於權重：指定為 (outC, inC, kH, kW)
# - 對於 bias：指定為 (N,)
array_shapes = {
    
    "cnn_conv_1_w": (32, 3, 3, 3),
    "cnn_conv_2_w": (32, 32, 3, 3),
    "cnn_conv_3_w": (32, 32, 3, 3),
    "cnn_conv_4_w": (32, 32, 3, 3),
    "cnn_conv_5_w": (32, 32, 3, 3),
    "cnn_conv_6_w": (32, 32, 3, 3),
    "cnn_conv_7_w": (32, 32, 3, 3),
    "cnn_conv_8_w": (32, 32, 3, 3),
    "cnn_conv_9_w": (32, 32, 3, 3),
    "cnn_conv_10_w": (1, 32, 1, 1), 
    "cnn_conv_1_b": (32,),
    "cnn_conv_2_b": (32,),
    "cnn_conv_3_b": (32,),
    "cnn_conv_4_b": (32,),
    "cnn_conv_5_b": (32,),
    "cnn_conv_6_b": (32,),
    "cnn_conv_7_b": (32,),
    "cnn_conv_8_b": (32,),
    "cnn_conv_9_b": (32,),
    "cnn_conv_10_b": (1,)
}

# === 3. 讀取整份檔案 ===
with open(input_file, "r") as f:
    content = f.read()
    print("cnn_conv_1_w" in content)  # 如果是 False，就是根本沒讀進去

def extract_array(name, text):
    pattern = re.compile(
        rf"{name}\s*\[[^\]]*\]\s*=\s*\{{(.*)\}}\s*;",
        re.DOTALL
    )

    match = pattern.search(text)
    if not match:
        raise ValueError(f"找不到陣列 {name}")
    values = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", match.group(1))]
    return values

def write_1d_array(name, data, f):
    f.write(f"static const float {name}[{len(data)}] = {{\n")
    for i in range(0, len(data), 8):
        line = ", ".join(f"{v:.6f}" for v in data[i:i+8])
        f.write(f"  {line},\n")
    f.write("};\n\n")

def write_4d_array(name, data, shape, f):
    outC, inC, kH, kW = shape
    arr = np.array(data).reshape((outC, inC, kH, kW))
    f.write(f"static const float {name}[{outC}][{inC}][{kH}][{kW}] = {{\n")
    for oc in range(outC):
        f.write("  {\n")
        for ic in range(inC):
            f.write("    {\n")
            for kh in range(kH):
                row = ", ".join(f"{arr[oc, ic, kh, kw]:.6f}" for kw in range(kW))
                f.write(f"      {{{row}}}" + (",\n" if kh < kH - 1 else "\n"))
            f.write("    }" + (",\n" if ic < inC - 1 else "\n"))
        f.write("  }" + (",\n" if oc < outC - 1 else "\n"))
    f.write("};\n\n")

# === 4. 轉換所有陣列 ===
with open(output_file, "w") as f:
    for name, shape in array_shapes.items():
        values = extract_array(name, content)
        if len(shape) == 1:
            write_1d_array(name, values, f)
        elif len(shape) == 4:
            expected = np.prod(shape)
            if len(values) != expected:
                raise ValueError(f"{name} 資料長度 {len(values)} 不符 shape {shape}")
            write_4d_array(name, values, shape, f)
        else:
            raise ValueError(f"不支援 shape: {shape}")
