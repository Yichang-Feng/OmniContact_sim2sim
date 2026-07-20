import os
import glob
import re

directory = '/home/feng/OmniContact_sim2sim/deploy_omnicontact/'
files = glob.glob(os.path.join(directory, 'deploy_omnicontact*.py'))

# Check if file should be modified
def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # 1. Add needs_odom_calibration = False
    content = re.sub(
        r'(def sync_object_state\(\):\n\s+if box_body_id < 0:\n\s+return\n)',
        r'\1        needs_odom_calibration = False\n',
        content,
        count=1
    )

    # 2. Set needs_odom_calibration = True for gt_pos
    # Because of print statements inside, it's better to match the exact assignment block
    content = re.sub(
        r'(state_cmd\.obj_pos = gt_pos\n\s+state_cmd\.obj_quat = d\.xquat\[box_body_id\]\.copy\(\)\n)',
        r'\1                    needs_odom_calibration = True\n',
        content
    )

    # 3. Set needs_odom_calibration = True for no vision
    content = re.sub(
        r'(state_cmd\.obj_pos = d\.xpos\[box_body_id\]\.copy\(\)\n\s+state_cmd\.obj_quat = d\.xquat\[box_body_id\]\.copy\(\)\n)',
        r'\1                needs_odom_calibration = True\n',
        content
    )

    # 4. Replace use_direct_rel_poses with needs_odom_calibration
    content = re.sub(
        r'if not getattr\(state_cmd, "use_direct_rel_poses", False\):',
        r'if needs_odom_calibration:',
        content
    )

    # 5. Fix LocoMode clearing logic
    loco_pattern = r'(if not has_entered_loco:\n\s+has_entered_loco = True\n)(\s+if real_robot is not None and hasattr\(real_robot, "subscribe_odom"\):\n\s+print\("\\n\[Odom\] 机器人已切换至高位站立/工作模式，开始订阅 ROS2 /lio/odom 里程计并重置校准锚点！"\)\n\s+real_robot\.subscribe_odom\("/lio/odom"\)\n\s+odom_calibration\["initial_pos_xy"\] = None\n\s+odom_calibration\["initial_pos_z"\] = None\n\s+odom_calibration\["initial_yaw_quat"\] = None\n)'
    loco_replacement = r'\1                                if state_cmd.skill_cmd == FSMCommand.LOCO or FSM_controller.cur_policy.name == FSMStateName.LOCOMODE:\n    \2                                        vision_cache["last_pos"] = None\n                                        vision_cache["last_quat"] = None\n                                        sync_robot_state()\n'
    content = re.sub(loco_pattern, loco_replacement, content)

    # 6. For deploy_omnicontact_stand_prepare_test.py, fix the 'H' key to 'Z' key (122, 90)
    if 'stand_prepare_test' in filepath:
        content = content.replace("104, 72", "122, 90")
        content = content.replace("键盘 H 键", "键盘 Z 键")
        content = content.replace("键盘按下 H", "键盘按下 Z")

    if original != content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Patched: {os.path.basename(filepath)}")
    else:
        print(f"No changes or already patched: {os.path.basename(filepath)}")

for file in files:
    patch_file(file)

print("Done patching.")
