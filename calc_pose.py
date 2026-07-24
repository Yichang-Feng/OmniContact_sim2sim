import numpy as np
from scipy.spatial.transform import Rotation as R

def print_euler(quat_list, label, fmt):
    if fmt == 'wxyz':
        w, x, y, z = quat_list
        r = R.from_quat([x, y, z, w])
    else:
        r = R.from_quat(quat_list)
    euler = r.as_euler('xyz', degrees=True)
    print(f"{label}: {euler}")

print("Assuming xyzw:")
print_euler([-0.9951, 0.0045, 0.0904, 0.0395], "Base Quat", 'xyzw')
print("Assuming wxyz:")
print_euler([-0.9951, 0.0045, 0.0904, 0.0395], "Base Quat", 'wxyz')

