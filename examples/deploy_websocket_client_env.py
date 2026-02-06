#!/usr/bin/env python
"""基于 WebSocket 的环境客户端，连接到 SAC 推理服务器（二进制传输）"""

import asyncio

import websockets
import msgpack
import msgpack_numpy
import numpy as np

import gymnasium as gym
import gym_hil


# 配置
SERVER_URI = "ws://127.0.0.1:8888"

async def main():
    """运行环境客户端"""

     # 创建环境
    env = gym.make("gym_hil/PandaArrangeBoxesMouse-v0",image_obs=True, render_mode="human", max_episode_steps=1000,use_gripper=True,gripper_penalty=0.0, x_step_size=1.0, y_step_size=1.0, z_step_size=0.2)

    # 连接到服务器
    print(f"Connecting to server: {SERVER_URI}")
    async with websockets.connect(SERVER_URI) as websocket:
        print("Connected to server")

        # 重置环境
        obs, info = env.reset()

        while True:
            try:
               # 序列化为二进制并发送
                message = {
                    "observation": obs,
                }
                await websocket.send(msgpack.packb(message, default=msgpack_numpy.encode))

                # 接收二进制数据并反序列化
                response = await websocket.recv()
                action_data = msgpack.unpackb(response, object_hook=msgpack_numpy.decode)
                action = action_data["action"]

                # 环境步骤（直接使用处理后的动作）
                obs, reward, terminated, truncated, info = env.step(action)

                # 重置环境
                if terminated or truncated:
                    obs, info = env.reset()
                    print("Environment reset")
            except websockets.exceptions.ConnectionClosed:
                print("Connection to server closed")
                break
            except Exception as e:
                print(f"Error: {e}", exc_info=True)
                break


if __name__ == "__main__":
    asyncio.run(main())

