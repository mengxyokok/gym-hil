#!/usr/bin/env python
"""使用MuJoCo交互式查看器，支持鼠标点选物体"""

import mujoco
import mujoco.viewer
from pathlib import Path
import glfw
import numpy as np

# 加载场景
xml_path = Path(__file__).parent.parent / "gym_hil" / "assets" / "arrange_boxes_scene.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)


# 创建GLFW窗口
glfw.init()
window = glfw.create_window(1200, 900, "Interactive Viewer", None, None)
glfw.make_context_current(window)

# 创建场景和上下文
scene = mujoco.MjvScene(model, maxgeom=1000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# 设置相机
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)

# 鼠标回调函数
selected_geom = -1

def mouse_button_callback(window, button, action, mods):
    global selected_geom
    
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        # 获取鼠标位置和窗口尺寸
        x, y = glfw.get_cursor_pos(window)
        width, height = glfw.get_window_size(window)
        
        # 转换为归一化坐标 [0,1]，注意y轴需要反转
        relx = x / width
        rely = 1.0 - (y / height)  # OpenGL坐标系y轴反转
        
        # 计算宽高比
        aspect = width / height
        
        # 使用mjv_select进行选择
        # 注意：所有数组参数必须是可写的，且形状正确
        # selpnt: [3, 1], geomid/flexid/skinid: [1, 1]
        selpnt = np.zeros((3, 1), dtype=np.float64)
        geomid_arr = np.array([[-1]], dtype=np.int32)
        flexid_arr = np.array([[-1]], dtype=np.int32)
        skinid_arr = np.array([[-1]], dtype=np.int32)
        
        mujoco.mjv_select(
            model, data, opt, aspect, relx, rely, scene, 
            selpnt, geomid_arr, flexid_arr, skinid_arr
        )
        
        geomid = geomid_arr[0, 0]
        if geomid >= 0:
            selected_geom = geomid
            body_id = model.geom_bodyid[selected_geom]
            
            # 获取名称
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, selected_geom)
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            
            # 获取位置
            geom_pos = data.geom_xpos[selected_geom].copy()
            
            # 获取geom类型
            geom_type = int(model.geom(selected_geom).type)  # 转换为Python int
            geom_types = {
                mujoco.mjtGeom.mjGEOM_PLANE: "PLANE",
                mujoco.mjtGeom.mjGEOM_BOX: "BOX",
                mujoco.mjtGeom.mjGEOM_SPHERE: "SPHERE",
                mujoco.mjtGeom.mjGEOM_CAPSULE: "CAPSULE",
                mujoco.mjtGeom.mjGEOM_CYLINDER: "CYLINDER",
                mujoco.mjtGeom.mjGEOM_MESH: "MESH",
            }
            geom_type_name = geom_types.get(geom_type, f"TYPE_{geom_type}")
            
            print(f"\n选中对象:")
            print(f"  几何体ID: {selected_geom}, 名称: {geom_name}")
            print(f"  身体ID: {body_id}, 名称: {body_name}")
            print(f"  位置: ({geom_pos[0]:.4f}, {geom_pos[1]:.4f}, {geom_pos[2]:.4f})")
            print(f"  类型: {geom_type_name}")
            print()
        else:
            selected_geom = -1
            print("未选中任何对象")

# 设置回调
glfw.set_mouse_button_callback(window, mouse_button_callback)

# 主循环
while not glfw.window_should_close(window):
    # 前向计算
    mujoco.mj_forward(model, data)
    
    # 更新场景
    perturb = mujoco.MjvPerturb()
    mujoco.mjv_updateScene(
        model, data, opt, perturb, cam,
        mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    
    # 渲染
    viewport = mujoco.MjrRect(0, 0, *glfw.get_window_size(window))
    mujoco.mjr_render(viewport, scene, context)
    
    # 交换缓冲区
    glfw.swap_buffers(window)
    glfw.poll_events()
    
    # 仿真步骤
    mujoco.mj_step(model, data)

glfw.terminate()