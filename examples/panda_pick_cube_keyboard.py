#!/usr/bin/env python

import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401

# 创建环境
env = gym.make("gym_hil/PandaPickCubeKeyboard-v0", render_mode="human")

# 创建dummy_action（键盘输入会自动覆盖）
action = np.zeros(env.action_space.shape[0], dtype=np.float32)


# 重置环境
obs, _ = env.reset()

try:
    while True:
        # action来自键盘输入（按Space激活干预模式后生效）
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
    print("测试结束")

