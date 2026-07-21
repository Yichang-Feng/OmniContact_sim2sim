import re

def parse_log_and_print_joints(filepath):
    print(f"\n--- Analyzing Joints {filepath} ---")
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
        m_joints = re.search(r'Q\(前6维腿部\)=\[([^\]]+)\]', block)
        
        if m_pos and m_joints:
            joints_str = m_joints.group(1)
            joints = [float(x.strip()) for x in joints_str.split(',')]
            data.append({
                'step': step,
                'mode': mode,
                'z': float(m_pos.group(1)),
                'joints': joints
            })
            
    # Find transition
    trans_idx = -1
    for i in range(1, len(data)):
        if data[i]['mode'] != data[i-1]['mode']:
            trans_idx = i
            break
            
    if trans_idx != -1:
        start_idx = max(0, trans_idx - 2)
        end_idx = min(len(data), trans_idx + 2)
        print(f"Transition at step {data[trans_idx]['step']}")
        for i in range(start_idx, end_idx):
            d = data[i]
            marker = "--> " if i == trans_idx else "    "
            j_str = ", ".join([f"{x:>6.3f}" for x in d['joints']])
            print(f"{marker}Step: {d['step']:<5} | Z: {d['z']:>6.4f} | Joints: [{j_str}]")
    else:
        print("No mode transition found.")

for f in [
    'object_pose_logging_stand_test.txt',
    'object_pose_logging_stand_test_real4.txt'
]:
    parse_log_and_print_joints(f)
