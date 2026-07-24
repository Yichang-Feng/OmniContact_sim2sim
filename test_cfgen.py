import numpy as np
import sys
sys.path.append('.')
from policy.omnicontact.CFgen_reference import plan_cfgen_reference
class MockPolicy:
    def __init__(self):
        self.task = "carrybox"
        self.goal_pos = np.array([1.2, 0.0, 0.0])
        self.robot = "g1"
        self.visual = False
        self.dt = 0.02
        self.box_dims = [0.4, 0.3, 0.2]
policy = MockPolicy()
fk_info = {
    "pelvis_p": np.array([0, 0, 0], dtype=np.float32),
    "pelvis_R": np.eye(3, dtype=np.float32),
    "obj_p": np.array([0.8, 0, 0.2], dtype=np.float32),
    "obj_R": np.eye(3, dtype=np.float32),
    "q": np.zeros(29, dtype=np.float32)
}
plan_cfgen_reference(policy, fk_info)
ref_phase = np.asarray(policy.ref_phase, dtype=np.int32).reshape(-1)
print(f"Total steps: {len(ref_phase)}")
unique_phases = np.unique(ref_phase)
print(f"Unique phases: {unique_phases}")
for p in unique_phases:
    idx = np.where(ref_phase == p)[0]
    print(f"Phase {p}: steps {idx[0]} to {idx[-1]} (count: {len(idx)})")
