#!/usr/bin/env python
"""Utility functions for printing scene object information."""

import mujoco


def print_scene_objects(model, data):
    """Print information about all objects in the loaded scene.
    
    Args:
        model: MuJoCo model object
        data: MuJoCo data object
    """
    try:
        print("\n" + "=" * 80)
        print("场景物体集合信息")
        print("=" * 80)
        
        # Print all bodies
        print(f"\n【Body 列表】 (共 {model.nbody} 个):")
        print("-" * 80)
        for i in range(model.nbody):
            try:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name == "":
                    body_name = f"body_{i}"
                
                # Get body position
                body_pos = data.xpos[i].copy()
                
                print(f"  [{i:3d}] {body_name:30s} | 位置: ({body_pos[0]:7.4f}, {body_pos[1]:7.4f}, {body_pos[2]:7.4f})")
            except:
                print(f"  [{i:3d}] body_{i} (无法获取详细信息)")
        
        # Print all geoms
        print(f"\n【Geom 列表】 (共 {model.ngeom} 个):")
        print("-" * 80)
        for i in range(model.ngeom):
            try:
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                if geom_name == "":
                    geom_name = f"geom_{i}"
                
                geom = model.geom(i)
                body_id = model.geom_bodyid[i]
                
                # Get geom type name
                geom_type_names = {
                    mujoco.mjtGeom.mjGEOM_PLANE: "PLANE",
                    mujoco.mjtGeom.mjGEOM_BOX: "BOX",
                    mujoco.mjtGeom.mjGEOM_SPHERE: "SPHERE",
                    mujoco.mjtGeom.mjGEOM_CAPSULE: "CAPSULE",
                    mujoco.mjtGeom.mjGEOM_CYLINDER: "CYLINDER",
                    mujoco.mjtGeom.mjGEOM_MESH: "MESH",
                }
                geom_type_name = geom_type_names.get(geom.type, f"TYPE_{geom.type}")
                
                # Get body name
                try:
                    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    if body_name == "":
                        body_name = f"body_{body_id}"
                except:
                    body_name = f"body_{body_id}"
                
                print(f"  [{i:3d}] {geom_name:30s} | 类型: {geom_type_name:10s} | 所属Body: {body_name}")
            except:
                print(f"  [{i:3d}] geom_{i} (无法获取详细信息)")
        
        # Print all sensors
        if model.nsensor > 0:
            print(f"\n【Sensor 列表】 (共 {model.nsensor} 个):")
            print("-" * 80)
            for i in range(model.nsensor):
                try:
                    sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                    if sensor_name == "":
                        sensor_name = f"sensor_{i}"
                    
                    sensor = model.sensor(i)
                    sensor_type_names = {
                        mujoco.mjtSensor.mjSENS_TOUCH: "TOUCH",
                        mujoco.mjtSensor.mjSENS_ACCELEROMETER: "ACCELEROMETER",
                        mujoco.mjtSensor.mjSENS_VELOCIMETER: "VELOCIMETER",
                        mujoco.mjtSensor.mjSENS_GYRO: "GYRO",
                        mujoco.mjtSensor.mjSENS_FORCE: "FORCE",
                        mujoco.mjtSensor.mjSENS_TORQUE: "TORQUE",
                        mujoco.mjtSensor.mjSENS_MAGNETOMETER: "MAGNETOMETER",
                        mujoco.mjtSensor.mjSENS_RANGEFINDER: "RANGEFINDER",
                        mujoco.mjtSensor.mjSENS_CAMERA: "CAMERA",
                        mujoco.mjtSensor.mjSENS_JOINTPOS: "JOINTPOS",
                        mujoco.mjtSensor.mjSENS_JOINTVEL: "JOINTVEL",
                        mujoco.mjtSensor.mjSENS_TENDONPOS: "TENDONPOS",
                        mujoco.mjtSensor.mjSENS_TENDONVEL: "TENDONVEL",
                        mujoco.mjtSensor.mjSENS_ACTUATORPOS: "ACTUATORPOS",
                        mujoco.mjtSensor.mjSENS_ACTUATORVEL: "ACTUATORVEL",
                        mujoco.mjtSensor.mjSENS_ACTUATORFRC: "ACTUATORFRC",
                        mujoco.mjtSensor.mjSENS_BALLQUAT: "BALLQUAT",
                        mujoco.mjtSensor.mjSENS_BALLANGVEL: "BALLANGVEL",
                        mujoco.mjtSensor.mjSENS_JOINTLIMITPOS: "JOINTLIMITPOS",
                        mujoco.mjtSensor.mjSENS_JOINTLIMITVEL: "JOINTLIMITVEL",
                        mujoco.mjtSensor.mjSENS_JOINTLIMITFRC: "JOINTLIMITFRC",
                        mujoco.mjtSensor.mjSENS_TENDONLIMITPOS: "TENDONLIMITPOS",
                        mujoco.mjtSensor.mjSENS_TENDONLIMITVEL: "TENDONLIMITVEL",
                        mujoco.mjtSensor.mjSENS_TENDONLIMITFRC: "TENDONLIMITFRC",
                        mujoco.mjtSensor.mjSENS_FRAMEPOS: "FRAMEPOS",
                        mujoco.mjtSensor.mjSENS_FRAMEXAXIS: "FRAMEXAXIS",
                        mujoco.mjtSensor.mjSENS_FRAMEYAXIS: "FRAMEYAXIS",
                        mujoco.mjtSensor.mjSENS_FRAMEZAXIS: "FRAMEZAXIS",
                        mujoco.mjtSensor.mjSENS_FRAMELINVEL: "FRAMELINVEL",
                        mujoco.mjtSensor.mjSENS_FRAMEANGVEL: "FRAMEANGVEL",
                        mujoco.mjtSensor.mjSENS_FRAMELINACC: "FRAMELINACC",
                        mujoco.mjtSensor.mjSENS_FRAMEANGACC: "FRAMEANGACC",
                        mujoco.mjtSensor.mjSENS_SUBTREECOM: "SUBTREECOM",
                        mujoco.mjtSensor.mjSENS_SUBTREELINVEL: "SUBTREELINVEL",
                        mujoco.mjtSensor.mjSENS_SUBTREEANGMOM: "SUBTREEANGMOM",
                    }
                    sensor_type_name = sensor_type_names.get(sensor.type, f"TYPE_{sensor.type}")
                    
                    print(f"  [{i:3d}] {sensor_name:30s} | 类型: {sensor_type_name}")
                except:
                    print(f"  [{i:3d}] sensor_{i} (无法获取详细信息)")
        
        # Print cameras
        if model.ncam > 0:
            print(f"\n【Camera 列表】 (共 {model.ncam} 个):")
            print("-" * 80)
            for i in range(model.ncam):
                try:
                    camera_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                    if camera_name == "":
                        camera_name = f"camera_{i}"
                    
                    camera = model.camera(i)
                    print(f"  [{i:3d}] {camera_name:30s} | fovy: {camera.fovy:.2f}°")
                except:
                    print(f"  [{i:3d}] camera_{i} (无法获取详细信息)")
        
        print("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        print(f"输出场景物体信息时出错: {e}")


def main():
    """Test function to demonstrate print_scene_objects."""
    from pathlib import Path
    
    # Define model path - get the assets directory
    # From examples/ directory, go up to project root, then to gym_hil/assets/
    project_root = Path(__file__).parent.parent
    assets_dir = project_root / "gym_hil" / "assets"
    
    # Load arrange_boxes_scene.xml only
    model_path = assets_dir / "arrange_boxes_scene.xml"
    
    if not model_path.exists():
        print(f"错误: 场景文件不存在: {model_path}")
        return
    
    try:
        print(f"加载场景文件: {model_path}")
        model = mujoco.MjModel.from_xml_path(str(model_path))
    except Exception as e:
        print(f"加载场景文件失败: {e}")
        return
    
    # Create data object
    data = mujoco.MjData(model)
    
    # Forward step to initialize positions
    mujoco.mj_forward(model, data)
    
    # Print scene objects
    print_scene_objects(model, data)


if __name__ == "__main__":
    main()

