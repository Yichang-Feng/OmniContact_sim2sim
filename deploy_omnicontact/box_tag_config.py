#!/usr/bin/env python3
"""
集中式箱子尺寸与 AprilTag 参数配置中心
===================================================
修改此文件中的参数后，执行:
    python deploy_omnicontact/box_tag_config.py
即可一键同步更新 MuJoCo XML 仿真模型与 YAML 配置文件！
"""

import os
import re

# =========================================================
# 1. 箱子半尺寸参数 (hx, hy, hz)，单位：米
#    例如: [0.125, 0.215, 0.185] 对应长 25cm, 宽 43cm, 高 37cm
# =========================================================
BOX_HALF_DIMS = [0.125, 0.215, 0.185]

# =========================================================
# 2. AprilTag 标签物理参数
#    TAG_SIZE: 黑色方框的有效物理边长 (米)，例如 10cm = 0.10
# =========================================================
TAG_SIZE = 0.10

# =========================================================
# 3. MuJoCo 仿真中标签贴图文件路径 (相对于 g1_description 目录)
#    例如: "../data/tag582_100.png" 或 "../data/tag36_11_00000.png"
# =========================================================
TAG_TEXTURE_FILE = "../data/tag582_100.png"

# =========================================================
# 4. 标签布局模式选择参数 (TAG_LAYOUT)
#    "1tag": 现有的顶部中心单标签方案
#    "4tag": 顶部四个角分布 4 个标签方案 (tag36_11_00001.png ~ tag36_11_00004.png)
# =========================================================
TAG_LAYOUT = "4tag"


def sync_all(target_layout=None):
    if target_layout is None:
        target_layout = TAG_LAYOUT

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    xml_path = os.path.join(root_dir, "g1_description", "omnicontact_carry_box.xml")
    yaml_path = os.path.join(root_dir, "deploy_omnicontact", "config", "mujoco.yaml")

    hx, hy, hz = BOX_HALF_DIMS
    size_str = f"{hx} {hy} {hz}"
    tag_half_s = TAG_SIZE / 2.0
    tag_pos_z = hz + 0.0005
    dx = hx - tag_half_s
    dy = hy - tag_half_s

    print(f"[BoxTagConfig] 正在同步箱子与标签配置 (方案: {target_layout})...")
    print(f"  -> 箱子半尺寸: {size_str} (实物全长 {hx*200:.1f} x {hy*200:.1f} x {hz*200:.1f} cm)")
    print(f"  -> AprilTag 边长: {TAG_SIZE*100:.1f} cm")
    if target_layout == "4tag":
        print(f"  -> 仿真贴图: 顶部四个角分布 tag36_11_00001.png ~ tag36_11_00004.png")
    else:
        print(f"  -> 仿真贴图: {TAG_TEXTURE_FILE}")

    # 1. 更新 XML 文件
    if os.path.exists(xml_path):
        with open(xml_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 更新 ghost_visual 尺寸
        content = re.sub(
            r'(<geom\s+type="box"\s+size=")[^"]*("\s+contype="0"\s+conaffinity="0"\s+group="1")',
            r'\g<1>' + size_str + r'\g<2>',
            content,
        )
        # 更新 box_geom 尺寸
        content = re.sub(
            r'(<geom\s+name="box_geom"\s+type="box"\s+size=")[^"]*(")',
            r'\g<1>' + size_str + r'\g<2>',
            content,
        )
        # 更新 box body 高度
        content = re.sub(
            r'(<body\s+name="box"\s+pos="1\s+0\s+)[^"]*(")',
            r'\g<1>' + f"{hz}" + r'\g<2>',
            content,
        )

        # 根据 target_layout 更新 asset 和 box 上的 geom
        if target_layout == "4tag":
            new_asset = (
                '        <texture name="apriltag_tex_1" type="2d" file="../data/tag36_11_00001.png" />\n'
                '        <material name="apriltag_mat_1" texture="apriltag_tex_1" specular="0" shininess="0" />\n'
                '        <texture name="apriltag_tex_2" type="2d" file="../data/tag36_11_00002.png" />\n'
                '        <material name="apriltag_mat_2" texture="apriltag_tex_2" specular="0" shininess="0" />\n'
                '        <texture name="apriltag_tex_3" type="2d" file="../data/tag36_11_00003.png" />\n'
                '        <material name="apriltag_mat_3" texture="apriltag_tex_3" specular="0" shininess="0" />\n'
                '        <texture name="apriltag_tex_4" type="2d" file="../data/tag36_11_00004.png" />\n'
                '        <material name="apriltag_mat_4" texture="apriltag_tex_4" specular="0" shininess="0" />'
            )
            new_tags = (
                f'            <geom name="tag_1" type="box" size="{tag_half_s:.4f} {tag_half_s:.4f} 0.0005" pos="{dx:.4f} {dy:.4f} {tag_pos_z:.4f}" material="apriltag_mat_1" mass="0.001" contype="0" conaffinity="0" />\n'
                f'            <geom name="tag_2" type="box" size="{tag_half_s:.4f} {tag_half_s:.4f} 0.0005" pos="{-dx:.4f} {dy:.4f} {tag_pos_z:.4f}" material="apriltag_mat_2" mass="0.001" contype="0" conaffinity="0" />\n'
                f'            <geom name="tag_3" type="box" size="{tag_half_s:.4f} {tag_half_s:.4f} 0.0005" pos="{-dx:.4f} {-dy:.4f} {tag_pos_z:.4f}" material="apriltag_mat_3" mass="0.001" contype="0" conaffinity="0" />\n'
                f'            <geom name="tag_4" type="box" size="{tag_half_s:.4f} {tag_half_s:.4f} 0.0005" pos="{dx:.4f} {-dy:.4f} {tag_pos_z:.4f}" material="apriltag_mat_4" mass="0.001" contype="0" conaffinity="0" />'
            )
        else:
            new_asset = f'        <texture name="apriltag_tex" type="2d" file="{TAG_TEXTURE_FILE}" />\n        <material name="apriltag_mat" texture="apriltag_tex" specular="0" shininess="0" />'
            new_tags = f'            <geom name="tag_0" type="box" size="{tag_half_s:.4f} {tag_half_s:.4f} 0.0005" pos="0 0 {tag_pos_z:.4f}" material="apriltag_mat" mass="0.001" contype="0" conaffinity="0" />'

        content = re.sub(
            r'(<material\s+name="groundplane"[^>]*>)\s*(.*?)\s*(</asset>)',
            r'\1\n' + new_asset + r'\n    \3',
            content,
            flags=re.DOTALL,
        )
        content = re.sub(
            r'(<geom\s+name="box_geom"\s+[^>]*/>)\s*(.*?)\s*(</body\s*>)',
            r'\1\n' + new_tags + r'\n        \3',
            content,
            flags=re.DOTALL,
        )

        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [✓] 已同步更新 MuJoCo XML: {xml_path}")
    else:
        print(f"  [!] 未找到 XML 文件: {xml_path}")

    # 2. 更新 YAML 文件
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()

        content = re.sub(
            r'(box_half_dims:\s*\[)[^\]]*(\])',
            r'\g<1>' + f"{hx}, {hy}, {hz}" + r'\g<2>',
            content,
        )
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [✓] 已同步更新配置文件: {yaml_path}")
    else:
        print(f"  [!] 未找到 YAML 文件: {yaml_path}")

    print("[BoxTagConfig] 全部同步完毕！\n")


def switch_layout(target_layout):
    """一键修改 box_tag_config.py 中的 TAG_LAYOUT 参数并同步项目相关文件"""
    global TAG_LAYOUT
    if target_layout not in ["1tag", "4tag"]:
        print(f"[!] 错误：不支持的布局模式 '{target_layout}'，仅支持 '1tag' 或 '4tag'")
        return
    file_path = os.path.abspath(__file__)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(
        r'(TAG_LAYOUT\s*=\s*")[^"]*(")',
        r'\g<1>' + target_layout + r'\g<2>',
        content,
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    TAG_LAYOUT = target_layout
    print(f"[BoxTagConfig] 已一键修改 box_tag_config.py 中的 TAG_LAYOUT 为 '{target_layout}'")
    sync_all(target_layout)


def switch_to_4_tags():
    """一键将相应文件从一个tag改变为4个"""
    switch_layout("4tag")


def switch_to_1_tag():
    """一键切回 1 个 tag 方案"""
    switch_layout("1tag")


if __name__ == "__main__":
    import sys
    if "--4tag" in sys.argv or "--layout=4tag" in sys.argv or ("--layout" in sys.argv and "4tag" in sys.argv):
        switch_to_4_tags()
    elif "--1tag" in sys.argv or "--layout=1tag" in sys.argv or ("--layout" in sys.argv and "1tag" in sys.argv):
        switch_to_1_tag()
    else:
        sync_all()
