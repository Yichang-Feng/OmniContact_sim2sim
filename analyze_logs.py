import re

def parse_log_and_print(filepath):
    print(f"\n--- Analyzing {filepath} ---")
    data = []
    
    with open(filepath, 'r') as f:
        content = f.read()
        
    blocks = content.split('=== Step:')
    for block in blocks[1:]:
        header_line = block.split('\n')[0]
        m_header = re.search(r'\s*(\d+).*Time:\s*([0-9.]+)s.*?Mode:\s*([^)]+)', header_line)
        if not m_header:
            continue
        step = int(m_header.group(1))
        mode = m_header.group(3).strip()
        
        m_pos = re.search(r'Base Pos.*?: \[[^,]+, [^,]+, ([0-9.-]+)\]', block)
        m_quat = re.search(r'Base Quat.*?: \[([0-9.-]+), ([0-9.-]+), ([0-9.-]+), ([0-9.-]+)\]', block)
        m_vel = re.search(r'LinVel=\[([0-9.-]+), ([0-9.-]+), ([0-9.-]+)\].*?AngVel=\[([0-9.-]+), ([0-9.-]+), ([0-9.-]+)\]', block)
        
        if m_pos and m_quat and m_vel:
            data.append({
                'step': step,
                'mode': mode,
                'z': float(m_pos.group(1)),
                'qx': float(m_quat.group(2)),
                'qy': float(m_quat.group(3)),
                'vz': float(m_vel.group(3)),
                'wx': float(m_vel.group(4)),
                'wy': float(m_vel.group(5)),
            })
            
    # Find transition
    trans_idx = -1
    for i in range(1, len(data)):
        if data[i]['mode'] != data[i-1]['mode']:
            trans_idx = i
            break
            
    if trans_idx != -1:
        start_idx = max(0, trans_idx - 5)
        end_idx = min(len(data), trans_idx + 10)
        print(f"Transition from {data[trans_idx-1]['mode']} to {data[trans_idx]['mode']} at step {data[trans_idx]['step']}")
        for i in range(start_idx, end_idx):
            d = data[i]
            marker = "--> " if i == trans_idx else "    "
            print(f"{marker}Step: {d['step']:<5} | Mode: {d['mode']:<15} | Z: {d['z']:>6.4f} | Qx: {d['qx']:>7.4f} | Qy: {d['qy']:>7.4f} | Vz: {d['vz']:>7.4f} | Wx: {d['wx']:>7.4f} | Wy: {d['wy']:>7.4f}")
    else:
        print("No mode transition found.")

for f in [
    'object_pose_logging_stand_test.txt',
    'object_pose_logging_stand_test_real4.txt',
    'object_pose_logging_stand_test_real5.txt'
]:
    parse_log_and_print(f)
