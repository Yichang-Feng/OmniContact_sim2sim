import sys
import re
import math
import numpy as np

def parse_log(filename):
    steps = []
    base_pos = []
    base_quat = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        step_match = re.search(r"=== Step: (\d+)", line)
        if step_match:
            steps.append(int(step_match.group(1)))
        if "里程计位置" in line and "Base Pos" in line:
            bracket_content = re.search(r"\[(.*?)\]", line)
            if bracket_content:
                vals = bracket_content.group(1).split(',')
                base_pos.append([float(v.strip()) for v in vals])
        if "里程计朝向" in line and "Base Quat" in line:
            bracket_content = re.search(r"\[(.*?)\]", line)
            if bracket_content:
                vals = bracket_content.group(1).split(',')
                base_quat.append([float(v.strip()) for v in vals])
    return steps, base_pos, base_quat

def calc_std(arr):
    if len(arr) == 0: return 0
    mean = sum(arr) / len(arr)
    return math.sqrt(sum((a - mean)**2 for a in arr) / len(arr))

filename = 'object_pose_logging_stand_test.txt'
steps, base_pos, base_quat = parse_log(filename)
if len(base_pos) > 0:
    print(f"\n--- {filename} ---")
    print(f"Num steps: {len(steps)}")
    px, py, pz = [p[0] for p in base_pos], [p[1] for p in base_pos], [p[2] for p in base_pos]
    qw, qx, qy, qz = [q[0] for q in base_quat], [q[1] for q in base_quat], [q[2] for q in base_quat], [q[3] for q in base_quat]
    
    print(f"Base Pos Std: X={calc_std(px):.4f}, Y={calc_std(py):.4f}, Z={calc_std(pz):.4f}")
    print(f"Base Pos Diff: X={max(px)-min(px):.4f}, Y={max(py)-min(py):.4f}, Z={max(pz)-min(pz):.4f}")
    
    print(f"Base Quat Std: W={calc_std(qw):.4f}, X={calc_std(qx):.4f}, Y={calc_std(qy):.4f}, Z={calc_std(qz):.4f}")
    print(f"Base Quat Diff: W={max(qw)-min(qw):.4f}, X={max(qx)-min(qx):.4f}, Y={max(qy)-min(qy):.4f}, Z={max(qz)-min(qz):.4f}")
else:
    print("No data found!")
