import numpy as np

def quat_apply(q, v):
    w, x, y, z = q
    v0, v1, v2 = v
    t0 = 2 * (y * v2 - z * v1)
    t1 = 2 * (z * v0 - x * v2)
    t2 = 2 * (x * v1 - y * v0)
    return np.array([
        v0 + w * t0 + y * t2 - z * t1,
        v1 + w * t1 + z * t0 - x * t2,
        v2 + w * t2 + x * t1 - y * t0
    ])
def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

# Step 2540
base_pos = np.array([0.0086, -0.0134, 0.7806])
base_quat = np.array([-0.9799, 0.0293, 0.1919, 0.0467])
upper_pos_rel = np.array([1.0553, 0.1426, -0.5471])
lower_pos_rel = np.array([1.1131, 0.0438, -0.5008])

# We know: upper_pos_rel = quat_apply(quat_conjugate(base_quat), obj_pos - base_pos)
# So obj_pos - base_pos = quat_apply(base_quat, upper_pos_rel)
obj_pos_minus_base = quat_apply(base_quat, upper_pos_rel)
obj_pos = base_pos + obj_pos_minus_base

print(f"Calculated obj_pos: {obj_pos}")

# We also know lower_pos_rel is yaw-only.
# So lower_pos_rel_Z = obj_pos[2] - torso_pos[2]
torso_pos_Z = obj_pos[2] - lower_pos_rel[2]
print(f"Calculated torso_pos_Z: {torso_pos_Z}")

print(f"base_pos_Z: {base_pos[2]}")
print(f"torso_pos_Z - base_pos_Z (should be positive ~0.044): {torso_pos_Z - base_pos[2]}")

