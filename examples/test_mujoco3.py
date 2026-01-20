#!/usr/bin/env python
"""使用MuJoCo交互式查看器，支持鼠标双击选物体"""

import mujoco
import mujoco.viewer
from pathlib import Path
import glfw
import numpy as np

# 加载场景
xml_path = Path(__file__).parent.parent / "gym_hil" / "assets" / "arrange_boxes_scene.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# 启动查看器（自动支持鼠标交互）
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 仿真循环
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        # 检查选中对象（最短实现）
        if viewer.perturb.select > 0:
            gid = viewer.perturb.select
            print(f"选中: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)} (ID:{gid})")