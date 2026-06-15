import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)

# 初始化姿态
data.qpos[:] = 0
data.qpos[2] = 0.9
data.qpos[3] = 1.0
data.qpos[4] = 0.0
data.qpos[5] = 0.0
data.qpos[6] = 0.0
data.qvel[:] = 0
data.ctrl[:] = 0

# 运动参数
walk_freq = 0.04
leg_amp = 0.15
arm_amp = 0.08
head_freq = 0.02   # 转头速度
head_amp = 25      # 转头幅度

with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running():
        dt = model.opt.timestep
        t += dt
        phase = t * walk_freq
        head_phase = t * head_freq

        # 原有走路逻辑
        data.ctrl[1] = np.sin(phase) * arm_amp
        data.ctrl[2] = np.sin(phase) * arm_amp * 0.4
        data.ctrl[8] = np.sin(phase) * leg_amp
        data.ctrl[9] = np.sin(phase) * leg_amp * 0.3

        data.ctrl[3] = np.sin(phase + np.pi) * arm_amp
        data.ctrl[4] = np.sin(phase + np.pi) * arm_amp * 0.4
        data.ctrl[5] = np.sin(phase + np.pi) * leg_amp
        data.ctrl[6] = np.sin(phase + np.pi) * leg_amp * 0.3

        # 新增：颈部左右转头
        data.ctrl[0] = np.sin(head_phase) * head_amp

        # 脚踝固定
        data.ctrl[7] = 0
        data.ctrl[10] = 0

        # 锁定姿态防倒地
        data.qpos[2] = 0.9
        data.qpos[3] = 1.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0
        data.qvel[:] = 0

        mujoco.mj_step(model, data)
        viewer.sync()