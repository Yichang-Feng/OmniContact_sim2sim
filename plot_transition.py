import os
import re
import matplotlib.pyplot as plt

files = [
    ('BEFORE (No Blend)', 'transition_log_BEFORE_no_blend.txt'),
    ('AFTER (With Blend)', 'transition_log_AFTER_with_blend.txt'),
    ('Sim (150_50)', 'transition_log_stand_test_sim_150_50.txt'),
    ('Real Machine', 'transition_log_stand_test.txt')
]

def parse_file_robust(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return {}
    with open(filename, 'r', encoding='utf-8') as f:
        full_text = f.read()
        
    stages_data = {}
    
    rows = full_text.split('m/s')
    for row in rows:
        if not row.strip():
            continue
            
        stage_m = re.search(r'S\s+(\d+)\s*\|', row)
        stage_idx = int(stage_m.group(1)) if stage_m else 0
        
        if stage_idx != 0:
            continue
            
        t_m = re.search(r'(\d+\.\d+)s', row)
        if not t_m:
            continue
            
        d_m = re.findall(r'Δ=\s*([+-]?\d+\.\d+)°', row)
        if len(d_m) < 2:
            continue
            
        v_all = re.findall(r'([+-]?\d+\.\d+)', row)
        if v_all:
            v = float(v_all[-1])
        else:
            continue
            
        if stage_idx not in stages_data:
            stages_data[stage_idx] = {'times': [], 'knee': [], 'hip': [], 'vel': []}
            
        stages_data[stage_idx]['times'].append(float(t_m.group(1)))
        stages_data[stage_idx]['knee'].append(float(d_m[0]))
        stages_data[stage_idx]['hip'].append(float(d_m[1]))
        stages_data[stage_idx]['vel'].append(v)
        
    return stages_data

all_datasets = []
for label, filename in files:
    stages_data = parse_file_robust(filename)
    if not stages_data or 0 not in stages_data:
        all_datasets.append((label, None))
    else:
        all_datasets.append((label, stages_data[0]))

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.suptitle('Transition Log Comparisons: ΔQ and Ankle Z Velocity (Stage 0 Only)', fontsize=20, y=0.98)

for idx, (label, data) in enumerate(all_datasets):
    ax_q = axes[0, idx]
    ax_v = axes[1, idx]
    
    if data:
        times = data['times']
        knee_deltas = data['knee']
        hip_deltas = data['hip']
        vel_z = data['vel']
        
        # User requested to remove the spike around 0.48s-0.50s in AFTER (With Blend)
        if 'AFTER' in label:
            valid_idx = [i for i, t in enumerate(times) if t < 0.48]
            times = [times[i] for i in valid_idx]
            knee_deltas = [knee_deltas[i] for i in valid_idx]
            hip_deltas = [hip_deltas[i] for i in valid_idx]
            vel_z = [vel_z[i] for i in valid_idx]
        
        ax_q.plot(times, knee_deltas, label='Left Knee ΔQ (°)', marker='o', markersize=3, color='blue')
        ax_q.plot(times, hip_deltas, label='Left Hip ΔQ (°)', marker='s', markersize=3, color='green')
        ax_q.set_title(f'[{label}]\nStep Difference ΔQ', fontsize=14)
        ax_q.set_xlabel('Time (s)', fontsize=12)
        ax_q.set_ylabel('ΔQ (°)', fontsize=12)
        ax_q.legend()
        ax_q.grid(True, linestyle='--', alpha=0.7)
        
        ax_v.plot(times, vel_z, label='Ankle Z Velocity (m/s)', marker='^', markersize=3, color='red')
        ax_v.set_title(f'[{label}]\nAnkle Z Velocity', fontsize=14)
        ax_v.set_xlabel('Time (s)', fontsize=12)
        ax_v.set_ylabel('Velocity (m/s)', fontsize=12)
        ax_v.legend()
        ax_v.grid(True, linestyle='--', alpha=0.7)
    else:
        ax_q.set_title(f'{label}\n(Data Not Found)')
        ax_v.set_title(f'{label}\n(Data Not Found)')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
out_path = '/home/feng/OmniContact_sim2sim/transition_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Successfully generated plot at {out_path}')
