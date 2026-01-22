#!/usr/bin/env python
"""使用鼠标控制Panda机器人拾取立方体的示例"""

import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401

# 创建环境（已包含鼠标控制）
env = gym.make("gym_hil/PandaArrangeBoxesMouse-v0", render_mode="human", max_episode_steps=600, x_step_size=1.0, y_step_size=1.0, z_step_size=0.2)

# 创建dummy_action（鼠标输入会自动覆盖）
action = np.zeros(env.action_space.shape[0], dtype=np.float32)

# 重置环境
obs, _ = env.reset()

try:
    while True:
        # action来自鼠标输入（按中键激活干预模式后生效）
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 检查是否成功
        if info.get("succeed", False):
            print("成功！方块已被抓起")
        
        # Episode结束时重置
        if terminated or truncated:
            print("Episode结束，重置环境")
            obs, _ = env.reset()
            
except KeyboardInterrupt:
    print("用户中断")
finally:
    env.close()
    print("环境关闭")

