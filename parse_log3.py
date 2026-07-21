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

steps, base_pos, base_quat = parse_log('object_pose_logging_stand_test_real3.txt')
if len(base_pos) > 0:
    print(f"\n--- object_pose_logging_stand_test_real3.txt ---")
    print(f"Num steps: {len(steps)}")
    pos = np.array(base_pos)
    quat = np.array(base_quat)
    print(f"Base Pos Std: X={np.std(pos[:,0]):.4f}, Y={np.std(pos[:,1]):.4f}, Z={np.std(pos[:,2]):.4f}")
    print(f"Base Pos Diff: X={np.max(pos[:,0])-np.min(pos[:,0]):.4f}, Y={np.max(pos[:,1])-np.min(pos[:,1]):.4f}, Z={np.max(pos[:,2])-np.min(pos[:,2]):.4f}")
    
    print(f"Base Quat Std: W={np.std(quat[:,0]):.4f}, X={np.std(quat[:,1]):.4f}, Y={np.std(quat[:,2]):.4f}, Z={np.std(quat[:,3]):.4f}")
    print(f"Base Quat Diff: W={np.max(quat[:,0])-np.min(quat[:,0]):.4f}, X={np.max(quat[:,1])-np.min(quat[:,1]):.4f}, Y={np.max(quat[:,2])-np.min(quat[:,2]):.4f}, Z={np.max(quat[:,3])-np.min(quat[:,3]):.4f}")
