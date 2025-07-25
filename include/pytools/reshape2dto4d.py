import numpy as np

def to_c_array(arr, name="cnn_conv_4_w"):
    outC, inC, kH, kW = arr.shape
    s = f"static const float {name}[{outC}][{inC}][{kH}][{kW}] = {{\n"
    for oc in range(outC):
        s += "  {\n"
        for ic in range(inC):
            s += "    {\n"
            for kh in range(kH):
                row = ", ".join(f"{arr[oc, ic, kh, kw]:.6f}" for kw in range(kW))
                s += f"      {{{row}}}"
                if kh != kH-1:
                    s += ",\n"
                else:
                    s += "\n"
            s += "    }"
            if ic != inC-1:
                s += ",\n"
            else:
                s += "\n"
        s += "  }"
        if oc != outC-1:
            s += ",\n"
        else:
            s += "\n"
    s += "};\n"
    return s

def append_to_header(arr, name, header_path):
    c_array_string = to_c_array(arr, name)
    with open(header_path, 'a') as f:  # 用 append 模式
        f.write('\n')  # 保險一行空行
        f.write(c_array_string)
    print(f"已經把 {name} 寫到 {header_path} 的最後面！")



arr1 = np.array([
[[-2,-26, 21, 36,-57, 41, 54,-32, 72,2, -102, -29,-15, 63, 10, 80,-18, 76,-25,-19, 19, 19,41,-12, 51,-66,-50, 66,-52,-40, 44,-58]
, [-6, 11,4, 31,-40, 45,-17,0, 46,9,-53, -43, 18, 50, 27,-13, 58, 18, 39, 64,-32, 31, 127,-31, 62,-28,-10, 13, 36,-32,5,2]
, [ -51, 27, -5, 30,-36, 60,-40,-20, 69, 25,-82, -32,-50, 40, 25,-36,-19, 83, 16, 27,-27,-10,91,-71, 55,-39, 44, -3, 45,-41, 32,-15]
, [ -10,-26,-15, 78,6, 33, 20,-26,-64, 19,-32, -34, 17,2,8, 63,-18, -9,-44,-42, 42, 32,12, 29,-34,-73,-73,104,-38,1, 66, 18]
, [ -11,2,-29, 81, 36,8,-29, -6,-79,-18, 16, -42, 12, 25, 21,4, 63,-45, 21, 34, 68, 16,28,4, 13,-28,-12, 93,-10, 22,5,7]
, [ -36, 40,-40, 81, 15, 38,-63,-18,-72, 15, 11, -26,-22,-14, -6,-13,-45,-20, 41, -2, 61,-45,29,-69,-47,-48, 17, 69, 43,5, 52, 44]
, [38,-36,-19, 59, 31, 23, 63,-30,8,-32, 48,36, 16,-30, 62, 14, -6,-24,-49,-47, -2, 37, -32, 27, 24,-58,-79,-25,-22, 46, -4,4]
, [-5, 13, 44, 47, 67, 51, 33,-37, -2,-50, 44,12, 22,5, 33,-64, 74,-87, -9, 12,-24, 31,-1, 12,7,-56,-19,-33, 51, 32,-36, 10]
, [ -13, 44, -9, 64, 51, 39, 18,-40, -5,-41, 30,36, 14,-21,-19,-74,-48,-75,-10,-48, -1,-18,-8,-30,-11,-18, 43,-73, 36, 38,-17, 21]
, [ -28,-34,-12,-52, 25,-30, 58, 50, 79, 22,-43, -55, 40,-41,-73, 83,-25, 27,-61,9,-47, 23, 1, 34, 33, 13,-48, 17,-59,-59,1, 11]
, [ -11, -3, 16,-53, 47, -5,6, 56, 63, 15,7, -48, 33,-30,-27,9, 49,-18, 26, 47,-51,-34,45, 10, 29, 45,8,9, 35,-62,-34,7]
, [ -48, 30,2,-41, 61,-25,-37, 78, 98, 11, 24, -40,8,-70,-36, 42,-56,-10, 26, 19,-33,-67,50,-70, -9, 46, 37, -1, 36,-79,-17, 54]
, [41,-30,-89,-16, 44,-76, 85, 79,-84,-43, 34,62, 55, 46,-12, 64, 15, 82,-68, -7, 77, 44, -13, 15, -2,-35,-27, -7,-79,-39, 39,1]
, [57, -2,-69,-43, 48,-91, 41, 60,-91,-24, 90,72, 78, 67, 27,-12, 67, 51, 25, 59, 51, 12,42, -5, 22,-11, 50,-41,0,-31, -5, 42]
, [26,8,-60,5, 54,-98, -5,105,-65,-14, 30,52, 17, 11,-51, 13,-48, 93,4, -5, 90,-48,-4,-70, -1, 28, 95,-61, 21,-14,1, 50]
, [30,-71, -1,2,-36,-92, 58, 60, 39,-37,4,50, 50,-45, 47,-52,-23,-15,-32,-63,-71, 49, -23, 11,-27,-40,-52,-21,-50, 93, 62,-30]
, [-3,-51, 60,4,-13,-88,5, 39,1,-41, -4,45, 42,-39, 32,-90, 58,-53, -5, 19,-41, 36, 1,-10, -3,7, 19,1, 41, 98, 18,7]
, [ -21,-42, 39, 44,-21, -111,-30, 47, 30,-21,-43,23,2,-74, -4,-96,-23,6,-10,-45,-49,-14,-2,-56,-46, -4, 55,-44, 30, 99, 51, 15]
, [ -55,-18, 44,-31,-23,0, -1,-30, -9,-11,-27,20, 17,4,-83, 95,7,-73,-83,-54,-33,8, -33, 50,-29, 46,-55, -5,-68,-16,7,-44]
, [ -30, 34, 67,-24, 22,-27,-25,-43,3, -5, 61,38, 19, 12,-43,2, 66,-79,-14, 20,-66,-25,84, -5, -5, 89, 33, 13, 15,-29,-36,-25]
, [ -48, 42, 54,5, 39,-21,-66,-30, -9,-36, 12,10, 18,-36,-73, 47,-60,-75, -6,-24,-71,-54,59,-44,-20, 75, 66,-22, 51, 11,-25,-12]
, [ -22,-13, 17,-67,-31, 48, -4,-49,-45, 88,1,52, -1,-54, 59, 51,-21, 64,-43, 25, 79, 71, -54, 33,-24, 45,-55,-24,-95,-61,9,-19]
, [-1, 65, 65,-61,-11, 47,-48,-53,-77, 46, 55,65, 22,-50,102, 18, 41,-18, 42, 68, 69, 25, 4,9,1, 80,4,-17, -6,-22,-66,1]
, [ -48, 72, 35,-52,4, 34,-87,-64,-75, 88,-22,63,6,-75, 39, 25,-51, 23, 42,5, 72,8,10,-66,-21, 63, 73,-36, 11,-46,-60, 40]
, [20,-53,-27,-42,-42, 62, 27,-24, 55, 55, 30, -57, 13, 71,8,-51, -1, 58,-43, 48,-25, -2, -50, 56, -5,-35,-47, 38,-60, 45,-12,-48]
, [55,-13, 30,-47,-13, 65,-32,-38, 67, 31, 51, -70,-23, 99,4,-71, 70, 20, 44, 95,-25,2, -33, 10, 24,9, 27,5, 12, 46,-58,-17]
, [ 5, -3, 38,-26,-61, 80,-60,-32, 71, 29,-21, -77,-18, 53,-39,-82,-49, 20, 66, 66, -3,-60, -35,-61,-13,3, 40,-19, 44, 80,-37,-42]]
]
)
arr1 = arr1.T.reshape(32, 3, 3, 3)
append_to_header(arr1, "cnn_conv_1_w", "converted_Weight1.h")