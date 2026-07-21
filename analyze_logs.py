import re

def parse_log(filename):
    data = {'step': [], 'mode': [], 'base_pos': [], 'lin_vel': [], 'ang_vel': []}
    
    with open(filename, 'r', encoding='utf-8') as f:
        current_mode = ""
        current_step = 0
        for line in f:
            m = re.search(r'=== Step: (\d+) .*? Mode: (\w+) ===', line)
            if m:
                current_step = int(m.group(1))
                current_mode = m.group(2)
            
            m = re.search(r'里程计位置 .*?: \[([-\.\d]+),\s*([-\.\d]+),\s*([-\.\d]+)\]', line)
            if m:
                pos = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
                
            m = re.search(r'里程计速度 .*? LinVel=\[([-\.\d]+),\s*([-\.\d]+),\s*([-\.\d]+)\].*?AngVel=\[([-\.\d]+),\s*([-\.\d]+),\s*([-\.\d]+)\]', line)
            if m:
                lin_vel = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
                ang_vel = [float(m.group(4)), float(m.group(5)), float(m.group(6))]
                
                data['step'].append(current_step)
                data['mode'].append(current_mode)
                data['base_pos'].append(pos)
                data['lin_vel'].append(lin_vel)
                data['ang_vel'].append(ang_vel)
                
    return data

def calc_variance(lst):
    if len(lst) < 2: return 0.0
    mean = sum(lst) / len(lst)
    return sum((x - mean) ** 2 for x in lst) / len(lst)

def calc_max_diff(lst):
    if len(lst) < 2: return 0.0
    return max(lst) - min(lst)

def extract_omnicontact(data):
    indices = [i for i, m in enumerate(data['mode']) if m and 'omnicontact' in m.lower()]
    if not indices:
        indices = list(range(len(data['mode'])))
    pos = [data['base_pos'][i] for i in indices]
    lin_vel = [data['lin_vel'][i] for i in indices]
    ang_vel = [data['ang_vel'][i] for i in indices]
    steps = [data['step'][i] for i in indices]
    return steps, pos, lin_vel, ang_vel

data2 = parse_log('object_pose_logging_stand_test_real2.txt')
data3 = parse_log('object_pose_logging_stand_test_real3.txt')

steps2, pos2, lin_vel2, ang_vel2 = extract_omnicontact(data2)
steps3, pos3, lin_vel3, ang_vel3 = extract_omnicontact(data3)

def print_stats(name, pos, lin_vel, ang_vel):
    if not pos:
        print(f"{name}: No data")
        return
        
    pos_x = [p[0] for p in pos]
    pos_y = [p[1] for p in pos]
    pos_z = [p[2] for p in pos]
    
    lin_x = [v[0] for v in lin_vel]
    lin_y = [v[1] for v in lin_vel]
    
    ang_roll = [v[0] for v in ang_vel]
    ang_pitch = [v[1] for v in ang_vel]
    
    print(f"=== {name} ===")
    print(f"Data points in omnicontact mode: {len(pos)}")
    print(f"Base Pos Z (Height) - Mean: {sum(pos_z)/len(pos_z):.4f}, Max Diff: {calc_max_diff(pos_z):.4f}")
    
    print("\n[ 前后晃动指标 (Front/Back) ]")
    print(f"  Pos X  - Variance: {calc_variance(pos_x):.6f}, Max Diff: {calc_max_diff(pos_x):.4f}")
    print(f"  Lin X  - Variance: {calc_variance(lin_x):.6f}, Max Diff: {calc_max_diff(lin_x):.4f}")
    print(f"  Pitch  - Variance: {calc_variance(ang_pitch):.6f}, Max Diff: {calc_max_diff(ang_pitch):.4f}")
    
    print("\n[ 左右晃动指标 (Left/Right) ]")
    print(f"  Pos Y  - Variance: {calc_variance(pos_y):.6f}, Max Diff: {calc_max_diff(pos_y):.4f}")
    print(f"  Lin Y  - Variance: {calc_variance(lin_y):.6f}, Max Diff: {calc_max_diff(lin_y):.4f}")
    print(f"  Roll   - Variance: {calc_variance(ang_roll):.6f}, Max Diff: {calc_max_diff(ang_roll):.4f}")
    print("")

print_stats("Log 2", pos2, lin_vel2, ang_vel2)
print_stats("Log 3", pos3, lin_vel3, ang_vel3)

