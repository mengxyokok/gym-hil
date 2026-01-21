#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import mujoco
import numpy as np


def load_controller_config(controller_name: str, config_path: str | None = None) -> dict:
    """
    Load controller configuration from a JSON file.

    Args:
        controller_name: Name of the controller to load.
        config_path: Path to the config file. If None, uses the package's default config.

    Returns:
        Dictionary containing the selected controller's configuration.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "controller_config.json"

    with open(config_path) as f:
        config = json.load(f)

    controller_config = config[controller_name] if controller_name in config else config["default"]

    if controller_name not in config:
        print(f"Controller {controller_name} not found in config. Using default configuration.")

    return controller_config


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def reset(self):
        """Reset the controller."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "no-op"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "success": False,
            "failure": False,
            "intervention": False,
            "rerecord": False,
        }
        self.listener = None

    def start(self):
        """Start the keyboard listener."""
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.ctrl_r:
                    self.open_gripper_command = True
                elif key == keyboard.Key.ctrl_l:
                    self.close_gripper_command = True
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = "success"
                elif key == keyboard.Key.esc:
                    self.key_states["failure"] = True
                    self.episode_end_status = "failure"
                elif key == keyboard.Key.space:
                    self.key_states["intervention"] = not self.key_states["intervention"]
                elif key == keyboard.Key.r:
                    self.key_states["rerecord"] = True
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.ctrl_r:
                    self.open_gripper_command = False
                elif key == keyboard.Key.ctrl_l:
                    self.close_gripper_command = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Right Ctrl and Left Ctrl: Open and close gripper")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  Space: Start/Stop Intervention")
        print("  ESC: Exit")

    def stop(self):
        """Stop the keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """Get the current movement deltas from keyboard state."""
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z

    def should_save(self):
        """Return True if Enter was pressed (save episode)."""
        return self.key_states["success"] or self.key_states["failure"]

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.key_states["intervention"]

    def reset(self):
        """Reset the controller."""
        for key in self.key_states:
            self.key_states[key] = False


class GamepadController(InputController):
    """Generate motion deltas from gamepad input."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01, deadzone=0.1, config_path=None):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False
        self.config_path = config_path
        self.controller_config = None

    def start(self):
        """Initialize pygame and the gamepad."""
        import pygame

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        joystick_name = self.joystick.get_name()
        print(f"Initialized gamepad: {joystick_name}")

        # Load controller configuration based on joystick name
        self.controller_config = load_controller_config(joystick_name, self.config_path)

        # Get button mappings from config
        buttons = self.controller_config.get("buttons", {})

        print("Gamepad controls:")
        print(f"  {buttons.get('rb', 'RB')} button: Intervention")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick (vertical): Move in Z axis")
        print(f"  {buttons.get('lt', 'LT')} button: Close gripper")
        print(f"  {buttons.get('rt', 'RT')} button: Open gripper")
        print(f"  {buttons.get('b', 'B')}/Circle button: Exit")
        print(f"  {buttons.get('y', 'Y')}/Triangle button: End episode with SUCCESS")
        print(f"  {buttons.get('a', 'A')}/Cross button: End episode with FAILURE")
        print(f"  {buttons.get('x', 'X')}/Square button: Rerecord episode")

    def stop(self):
        """Clean up pygame resources."""
        import pygame

        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Process pygame events to get fresh gamepad readings."""
        import pygame

        # Get button mappings from config
        buttons = self.controller_config.get("buttons", {})
        y_button = buttons.get("y", 3)  # Default to 3 if not found
        a_button = buttons.get("a", 0)  # Default to 0 if not found (Logitech F310)
        x_button = buttons.get("x", 2)  # Default to 2 if not found (Logitech F310)
        lt_button = buttons.get("lt", 6)  # Default to 6 if not found
        rt_button = buttons.get("rt", 7)  # Default to 7 if not found
        rb_button = buttons.get("rb", 5)  # Default to 5 if not found

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == y_button:
                    self.episode_end_status = "success"
                elif event.button == a_button:
                    self.episode_end_status = "failure"
                elif event.button == x_button:
                    self.episode_end_status = "rerecord_episode"
                elif event.button == lt_button:
                    self.close_gripper_command = True
                elif event.button == rt_button:
                    self.open_gripper_command = True

            # Reset episode status on button release
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [x_button, a_button, y_button]:
                    self.episode_end_status = None
                elif event.button == lt_button:
                    self.close_gripper_command = False
                elif event.button == rt_button:
                    self.open_gripper_command = False

            # Check for RB button for intervention flag
            if self.joystick.get_button(rb_button):
                self.intervention_flag = True
            else:
                self.intervention_flag = False

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        import pygame

        try:
            # Get axis mappings from config
            axes = self.controller_config.get("axes", {})
            axis_inversion = self.controller_config.get("axis_inversion", {})

            # Get axis indices from config (with defaults if not found)
            left_x_axis = axes.get("left_x", 0)
            left_y_axis = axes.get("left_y", 1)
            right_y_axis = axes.get("right_y", 3)

            # Get axis inversion settings (with defaults if not found)
            invert_left_x = axis_inversion.get("left_x", False)
            invert_left_y = axis_inversion.get("left_y", True)
            invert_right_y = axis_inversion.get("right_y", True)

            # Read joystick axes
            x_input = self.joystick.get_axis(left_x_axis)  # Left/Right
            y_input = self.joystick.get_axis(left_y_axis)  # Up/Down
            z_input = self.joystick.get_axis(right_y_axis)  # Up/Down for Z

            # Apply deadzone to avoid drift
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # Apply inversion if configured
            if invert_left_x:
                x_input = -x_input
            if invert_left_y:
                y_input = -y_input
            if invert_right_y:
                z_input = -z_input

            # Calculate deltas
            delta_x = y_input * self.y_step_size  # Forward/backward
            delta_y = x_input * self.x_step_size  # Left/right
            delta_z = z_input * self.z_step_size  # Up/down

            return delta_x, delta_y, delta_z

        except pygame.error:
            print("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI."""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        """
        Initialize the HID gamepad controller.

        Args:
            step_size: Base movement step size in meters
            z_scale: Scaling factor for Z-axis movement
            deadzone: Joystick deadzone to prevent drift
        """
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None

        # Movement values (normalized from -1.0 to 1.0)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        # Button states
        self.buttons = {}
        self.quit_requested = False
        self.save_requested = False

    def find_device(self):
        """Look for the gamepad device by vendor and product ID."""
        import hid

        devices = hid.enumerate()
        for device in devices:
            device_name = device["product_string"]
            if any(controller in device_name for controller in ["Logitech", "Xbox", "PS4", "PS5"]):
                return device

        print("No gamepad found, check the connection and the product string in HID to add your gamepad")
        return None

    def start(self):
        """Connect to the gamepad using HIDAPI."""
        import hid

        self.device_info = self.find_device()
        if not self.device_info:
            self.running = False
            return

        try:
            print(f"Connecting to gamepad at path: {self.device_info['path']}")
            self.device = hid.device()
            self.device.open_path(self.device_info["path"])
            self.device.set_nonblocking(1)

            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            print(f"Connected to {manufacturer} {product}")

            print("Gamepad controls (HID mode):")
            print("  Left analog stick: Move in X-Y plane")
            print("  Right analog stick: Move in Z axis (vertical)")
            print("  Button 1/B/Circle: Exit")
            print("  Button 2/A/Cross: End episode with SUCCESS")
            print("  Button 3/X/Square: End episode with FAILURE")

        except OSError as e:
            print(f"Error opening gamepad: {e}")
            print("You might need to run this with sudo/admin privileges on some systems")
            self.running = False

    def stop(self):
        """Close the HID device connection."""
        if self.device:
            self.device.close()
            self.device = None

    def update(self):
        """
        Read and process the latest gamepad data.
        Due to an issue with the HIDAPI, we need to read the read the device several times in order to get a stable reading
        """
        for _ in range(10):
            self._update()

    def _update(self):
        """Read and process the latest gamepad data."""
        if not self.device or not self.running:
            return

        try:
            # Read data from the gamepad
            data = self.device.read(64)
            # Interpret gamepad data - this will vary by controller model
            # These offsets are for the Logitech RumblePad 2
            if data and len(data) >= 8:
                # Normalize joystick values from 0-255 to -1.0-1.0
                self.left_x = (data[1] - 128) / 128.0
                self.left_y = (data[2] - 128) / 128.0
                self.right_x = (data[3] - 128) / 128.0
                self.right_y = (data[4] - 128) / 128.0

                # Apply deadzone
                self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
                self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
                self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
                self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

                # Parse button states (byte 5 in the Logitech RumblePad 2)
                buttons = data[5]

                # Check if RB is pressed then the intervention flag should be set
                self.intervention_flag = data[6] in [2, 6, 10, 14]

                # Check if RT is pressed
                self.open_gripper_command = data[6] in [8, 10, 12]

                # Check if LT is pressed
                self.close_gripper_command = data[6] in [4, 6, 12]

                # Check if Y/Triangle button (bit 7) is pressed for saving
                # Check if X/Square button (bit 5) is pressed for failure
                # Check if A/Cross button (bit 4) is pressed for rerecording
                if buttons & 1 << 7:
                    self.episode_end_status = "success"
                elif buttons & 1 << 5:
                    self.episode_end_status = "failure"
                elif buttons & 1 << 4:
                    self.episode_end_status = "rerecord_episode"
                else:
                    self.episode_end_status = None

        except OSError as e:
            print(f"Error reading from gamepad: {e}")

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        # Calculate deltas - invert as needed based on controller orientation
        delta_x = -self.left_y * self.x_step_size  # Forward/backward
        delta_y = -self.left_x * self.y_step_size  # Left/right
        delta_z = -self.right_y * self.z_step_size  # Up/down

        return delta_x, delta_y, delta_z

    def should_quit(self):
        """Return True if quit button was pressed."""
        return self.quit_requested

    def should_save(self):
        """Return True if save button was pressed."""
        return self.save_requested


class MouseController(InputController):
    """Generate motion deltas from mouse input using pynput."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01, sensitivity=0.001):
        """
        Initialize the mouse controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
            sensitivity: Mouse movement sensitivity (converts pixel movement to meters)
        """
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.sensitivity = sensitivity
        self.listener = None
        self.last_mouse_pos = None
        self.gripper_state = False  # False = closed, True = open
        self.right_click_pending = False
        self.middle_click_pending = False
        self.current_mouse_pos = None
        self.left_click_pending = False
        self.click_movement_delta = None  # Movement delta from click
        self.env = None  # Reference to environment for getting gripper position
        self.target_object_pos = None  # Target object position for continuous movement
        self.is_moving_to_target = False  # Flag to indicate continuous movement
        self.side_button_up_pressed = False  # Side button up (x2) pressed
        self.side_button_down_pressed = False  # Side button down (x1) pressed
        self.last_selected_body_id = -1  # 上一次选中的geom ID，用于检测变化
        self.is_picking_object = True  # 是否拾取物体模式

    def start(self):
        """Start the mouse listener."""
        from pynput import mouse

        def on_move(x, y):
            """Handle mouse movement."""
            # Update current mouse position
            self.current_mouse_pos = (x, y)

        def on_click(x, y, button, pressed):
            """Handle mouse button clicks."""
            if button == mouse.Button.middle:
                # Middle button: toggle intervention on/off
                if pressed and not self.middle_click_pending:
                    self.intervention_flag = not self.intervention_flag
                    if self.intervention_flag:
                        print("✓ 干预模式已开启")
                    else:
                        print("✗ 干预模式已关闭")
                    self.middle_click_pending = True
                elif not pressed:
                    self.middle_click_pending = False
            elif button == mouse.Button.left:
                # Left button: click and hold to move end effector towards clicked object continuously
                if pressed and not self.left_click_pending:
                    # Get target object position (simplified: use block position)
                    if self.is_picking_object:
                        target_pos = self._get_picked_object_position()
                    else:
                        target_pos = self._get_clicked_object_position(x, y)
                    
                    if target_pos is not None:
                        # Store target position for continuous movement
                        self.target_object_pos = target_pos.copy()
                        self.is_moving_to_target = True
                        self.left_click_pending = True

                    else:
                        # No target found, stop movement
                        self.is_moving_to_target = False
                        self.target_object_pos = None
                        self.left_click_pending = True
                elif not pressed:
                    # Stop continuous movement when button is released
                    self.left_click_pending = False
                    self.is_moving_to_target = False
                    self.target_object_pos = None
                    self.click_movement_delta = None
            elif button == mouse.Button.right:
                # Right button: toggle gripper
                if pressed and not self.right_click_pending:
                    self.gripper_state = not self.gripper_state
                    self.right_click_pending = True
                    if self.gripper_state:
                        self.open_gripper_command = True
                        self.close_gripper_command = False
                    else:
                        self.open_gripper_command = False
                        self.close_gripper_command = True
                elif not pressed:
                    self.right_click_pending = False
                    # Reset gripper commands after a short delay
                    self.open_gripper_command = False
                    self.close_gripper_command = False
            elif button == mouse.Button.button9:
                # Side button up (forward/up arrow, typically button9): move end effector up
                if pressed:
                    self.side_button_up_pressed = True
                else:
                    self.side_button_up_pressed = False
            elif button == mouse.Button.button8:
                # Side button down (back/down arrow, typically button8): move end effector down
                if pressed:
                    self.side_button_down_pressed = True
                else:
                    self.side_button_down_pressed = False

        self.listener = mouse.Listener(
            on_move=on_move,
            on_click=on_click
        )
        self.listener.start()

        print("鼠标控制:")
        print("  鼠标中键按下: 切换干预模式开启/关闭")
        if self.is_picking_object:
            print("  鼠标左键双击: 选择目标物体")
            print("  鼠标左键按下: 控制末端移动到目标物体（连续移动）")
        else:
            print("  鼠标左键单击: 控制末端移动到目标物体")
            print("  鼠标左键按下: 控制末端移动到目标物体（连续移动）")
        print("  鼠标右键单击: 切换夹爪开合")
        print("  鼠标侧键向上(前进键): 末端持续垂直向上移动")
        print("  鼠标侧键向下(后退键): 末端持续垂直向下移动")

    def stop(self):
        """Stop the mouse listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()
    
    def set_env(self, env):
        """Set the environment instance for getting gripper position."""
        self.env = env
    
    def _get_gripper_3d_position(self):
        """Get gripper center position in 3D world space.
        
        Returns:
            numpy array of (x, y, z) position, or None if unavailable.
        """
        if self.env is None:
            return None
        
        try:
            unwrapped = self.env.unwrapped
            if not hasattr(unwrapped, '_data'):
                return None
            
            # Get gripper 3D position from sensor
            gripper_pos = unwrapped._data.sensor("2f85/pinch_pos").data.copy()
            return gripper_pos
        except Exception:
            return None
    

    def _get_clicked_object_position(self, screen_x, screen_y):
        """Get the 3D position of the object at the clicked screen coordinates.
        
        Args:
            screen_x: Screen X coordinate (pixels)
            screen_y: Screen Y coordinate (pixels)
            
        Returns:
            numpy array of (x, y, z) position, or None if unavailable.
        """
        pass


    def _get_picked_object_position(self):
        """Get the 3D position of the picked object.
 
        Returns:
            numpy array of (x, y, z) position, or None if unavailable.
        """
        if self.env is None:
            return None
        
        # 获取viewer
        viewer =  self.env.env.env.env.env._viewer
        unwrapped = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        
        # 获取model和data
        model = unwrapped.model if hasattr(unwrapped, 'model') else (unwrapped._model if hasattr(unwrapped, '_model') else None)
        data = unwrapped.data if hasattr(unwrapped, 'data') else (unwrapped._data if hasattr(unwrapped, '_data') else None)
        
        if viewer is None or model is None or data is None:
            return None
        
        # 检查perturb.select
        if hasattr(viewer, 'perturb') and viewer.perturb.select > 0:
            body_id = viewer.perturb.select
            
            # 只在选中物体变化时输出信息
            if body_id != self.last_selected_body_id:
                self.last_selected_body_id = body_id
                body_pos = data.xpos[body_id].copy()
                print(f"目标对象: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)} (ID:{body_id})")
                print(f"目标位置: ({body_pos[0]:.4f}, {body_pos[1]:.4f}, {body_pos[2]:.4f})")
            else:
                # 如果body_id没有变化，直接返回位置（不重复输出信息）
                try:
                    body_pos = data.xpos[body_id].copy()
                    return body_pos
                except:
                    return None
        else:
            # 如果没有选中对象，重置last_selected_body_id
            if self.last_selected_body_id != -1:
                self.last_selected_body_id = -1
        
        return None

    def update(self):
        """Update controller state - call this once per frame."""
        # Update last position for delta calculation if we have current position
        if self.current_mouse_pos is not None:
            if self.last_mouse_pos is None:
                self.last_mouse_pos = self.current_mouse_pos

    def get_deltas(self):
        """Get the current movement deltas from mouse state."""
        # Initialize deltas
        delta_x, delta_y, delta_z = 0.0, 0.0, 0.0
        
        # Check for continuous movement towards target object
        if self.is_moving_to_target and self.target_object_pos is not None:
            # Get current gripper position
            gripper_pos = self._get_gripper_3d_position()
            
            if gripper_pos is not None:
                # Calculate direction vector from gripper to target
                direction = self.target_object_pos - gripper_pos
                
                # Normalize and scale by step size
                distance = np.linalg.norm(direction)
                if distance > 0.001:  # Avoid division by zero
                    # Normalize direction
                    direction_normalized = direction / distance
                    
                    # Apply step size (move a fixed distance towards target per frame)
                    step_distance = min(distance, 0.02)  # Max step of 2cm per frame
                    movement = direction_normalized * step_distance
                    
                    delta_x = movement[0]
                    delta_y = movement[1]
                    # Z movement from target following
                    delta_z = movement[2]
                else:
                    # Already at target, stop moving
                    self.is_moving_to_target = False
                    self.target_object_pos = None
        
        # Add side button Z movement (vertical movement)
        side_button_step = 0.02  # Movement step per frame when side button is pressed
        if self.side_button_up_pressed:
            delta_z += side_button_step * self.z_step_size  # Move up (positive Z)
        if self.side_button_down_pressed:
            delta_z -= side_button_step * self.z_step_size  # Move down (negative Z)
        
        # If we have movement from target following, return it
        if abs(delta_x) > 0.0001 or abs(delta_y) > 0.0001 or abs(delta_z) > 0.0001:
            return delta_x, delta_y, delta_z
        
        # Check for click-based movement (one-time)
        if self.click_movement_delta is not None:
            delta_x, delta_y, delta_z = self.click_movement_delta
            # Reset after one frame to make it a single-step movement
            self.click_movement_delta = None
            return delta_x, delta_y, delta_z
        
        # No movement
        return 0.0, 0.0, 0.0

    def should_intervene(self):
        """Return True if intervention flag was set (middle button pressed)."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command based on mouse right click."""
        if self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"
        else:
            return "no-op"

    def reset(self):
        """Reset the controller."""
        self.last_mouse_pos = None
        self.current_mouse_pos = None
        self.intervention_flag = False
        self.right_click_pending = False
        self.left_click_pending = False
        self.middle_click_pending = False
        self.open_gripper_command = False
        self.close_gripper_command = False
        self.click_movement_delta = None
        self.target_object_pos = None
        self.is_moving_to_target = False
        self.side_button_up_pressed = False
        self.side_button_down_pressed = False
