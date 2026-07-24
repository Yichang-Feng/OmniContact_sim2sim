import time
import math
import re
import numpy as np
import mujoco
import mujoco.viewer

def quat_to_yaw(q):
    w, x, y, z = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def yaw_to_mat(yaw):
    return np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])

def parse_log(filename):
    steps = []
    
    with open(filename, 'r') as f:
        content = f.read()
        
    blocks = content.split('=== Step: ')
    
    for block in blocks[1:]:
        step_data = {}
        
        # parse step num
        m_step = re.search(r'^(\d+)', block)
        if not m_step: continue
        step_data['step'] = int(m_step.group(1))
        
        # Base Pos
        m_base_pos = re.search(r'Base Pos .*?: \[([^\]]+)\]', block)
        if m_base_pos:
            step_data['base_pos'] = np.array([float(x) for x in m_base_pos.group(1).split(',')])
            
        m_base_quat = re.search(r'Base Quat .*?: \[([^\]]+)\]', block)
        if m_base_quat:
            step_data['base_quat'] = np.array([float(x) for x in m_base_quat.group(1).split(',')])
            
        # Object Rel
        m_obj_pos = re.search(r'物体相对位置 \(Pos\): \[([^\]]+)\]', block)
        if m_obj_pos:
            step_data['obj_rel_pos'] = np.array([float(x) for x in m_obj_pos.group(1).split(',')])
            
        m_obj_quat = re.search(r'物体相对姿态 \(Quat\): \[([^\]]+)\]', block)
        if m_obj_quat:
            step_data['obj_rel_quat'] = np.array([float(x) for x in m_obj_quat.group(1).split(',')])
            
        m_box_dim = re.search(r'物体边界尺寸 \(Box Dims\): \[([^\]]+)\]', block)
        if m_box_dim:
            step_data['box_dims'] = np.array([float(x) for x in m_box_dim.group(1).split(',')])
            
        # Ref Traj
        m_ref = re.search(r'实时参考追踪目标 \(Ref Traj\): Base: \[([^\]]+)\] \| LWrist: \[([^\]]+)\] \| RWrist: \[([^\]]+)\]', block)
        if m_ref:
            step_data['ref_base'] = np.array([float(x) for x in m_ref.group(1).split(',')])
            step_data['ref_lwrist'] = np.array([float(x) for x in m_ref.group(2).split(',')])
            step_data['ref_rwrist'] = np.array([float(x) for x in m_ref.group(3).split(',')])
            
        steps.append(step_data)
        
    return steps

def add_visual_geom(viewer, pos, size, type, color):
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        type=type,
        size=size,
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=color
    )
    viewer.user_scn.ngeom += 1

def main():
    filename = 'object_pose_logging_stand_test_real5.txt'
    steps = parse_log(filename)
    if not steps:
        print("No valid steps found.")
        return
        
    model = mujoco.MjModel.from_xml_path('g1_description/g1_29dof.xml')
    data = mujoco.MjData(model)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90
        
        idx = 0
        while viewer.is_running():
            step_data = steps[idx]
            
            # Set robot base position (assume floating base in xml)
            if 'base_pos' in step_data and 'base_quat' in step_data:
                try:
                    data.qpos[:3] = step_data['base_pos']
                    data.qpos[3:7] = step_data['base_quat']
                except Exception as e:
                    pass
            mujoco.mj_forward(model, data)
            
            viewer.user_scn.ngeom = 0
            
            base_pos = step_data.get('base_pos', np.zeros(3))
            base_quat = step_data.get('base_quat', np.array([1,0,0,0]))
            yaw = quat_to_yaw(base_quat)
            R = yaw_to_mat(yaw)
            
            # Draw object
            if 'obj_rel_pos' in step_data and 'box_dims' in step_data:
                obj_pos = base_pos + R @ step_data['obj_rel_pos']
                box_size = step_data['box_dims'] / 2.0
                add_visual_geom(viewer, obj_pos, box_size, mujoco.mjtGeom.mjGEOM_BOX, [1, 0.5, 0, 0.5])
                
            # Draw Ref Traj
            if 'ref_base' in step_data:
                ref_base = base_pos + R @ step_data['ref_base']
                add_visual_geom(viewer, ref_base, [0.05, 0, 0], mujoco.mjtGeom.mjGEOM_SPHERE, [0, 0, 1, 0.8])
                
                ref_lwrist = base_pos + R @ step_data['ref_lwrist']
                add_visual_geom(viewer, ref_lwrist, [0.05, 0, 0], mujoco.mjtGeom.mjGEOM_SPHERE, [0, 1, 0, 0.8])
                
                ref_rwrist = base_pos + R @ step_data['ref_rwrist']
                add_visual_geom(viewer, ref_rwrist, [0.05, 0, 0], mujoco.mjtGeom.mjGEOM_SPHERE, [1, 0, 0, 0.8])
                
            viewer.sync()
            
            idx = (idx + 1) % len(steps)
            time.sleep(0.5)

if __name__ == "__main__":
    main()
