#!/usr/bin/env python

import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401

if __name__ == "__main__":
    # 创建环境
    env = gym.make("gym_hil/PandaPickCubeKeyboard-v0", render_mode="human")
    
    # 创建dummy_action（键盘控制环境使用，实际控制通过键盘输入）
    dummy_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    if len(dummy_action) >= 4:
        dummy_action[-1] = 1  # 设置gripper动作为"保持"
    
    # 重置环境
    obs, _ = env.reset()
    
    try:
        while True:
            # 使用dummy_action进行step
            obs, reward, terminated, truncated, info = env.step(dummy_action)
            
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

