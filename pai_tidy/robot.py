"""Franka Panda control: damped-least-squares IK + min-jerk Cartesian trajectories."""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

TCP_OFFSET = 0.1034  # hand origin -> fingertip pad centre, along hand z
OBSERVE_Q = np.array([0.0, -1.7, 0.0, -3.0, 0.0, 1.6, 0.785])
HOME_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785])
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0
GRIPPER_RELEASE = 175.0  # ~5.5 cm opening: frees the item without touching tray walls


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def down_rotation(yaw: float) -> np.ndarray:
    """Gripper pointing down, fingers closing along (cos yaw, sin yaw)."""
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.column_stack([np.cross(y, z), y, z])


class Robot:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, fps: int = 25,
                 on_frame: Callable[[], None] | None = None):
        self.m, self.d = model, data
        self.ik_data = mujoco.MjData(model)
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.lo = model.jnt_range[:7, 0].copy()
        self.hi = model.jnt_range[:7, 1].copy()
        self.frame_dt = 1.0 / fps
        self.next_frame = 0.0
        self.on_frame = on_frame
        self.ctrl_steps = max(1, int(round(0.01 / model.opt.timestep)))
        self.kp = model.actuator_gainprm[:7, 0].copy()
        # The Menagerie gripper servo (100 N/m) squeezes with only ~2 N; items slip out when the arm
        # decelerates. Scale gain/bias together so ctrl semantics stay the same (~12 N grip).
        g = 6.0
        model.actuator_gainprm[7, 0] *= g
        model.actuator_biasprm[7, 1:3] *= g
        self.q_des = OBSERVE_Q.copy()

    # ---------------------------------------------------------------- simulation
    def reset(self, q: np.ndarray = OBSERVE_Q):
        self.d.qpos[:7] = q
        self.d.qpos[7:9] = 0.04
        self.d.ctrl[7] = GRIPPER_OPEN
        mujoco.mj_forward(self.m, self.d)
        self.set_arm(q)

    def set_arm(self, q: np.ndarray):
        """Position command with gravity/Coriolis feed-forward (the servos alone sag by ~5 mm)."""
        self.q_des = np.asarray(q, float).copy()
        self.d.ctrl[:7] = self.q_des + self.d.qfrc_bias[:7] / self.kp

    def step(self, n: int = 1):
        for _ in range(n):
            self.d.ctrl[:7] = self.q_des + self.d.qfrc_bias[:7] / self.kp
            mujoco.mj_step(self.m, self.d)
            if self.on_frame is not None and self.d.time >= self.next_frame:
                self.next_frame += self.frame_dt
                self.on_frame()

    def wait(self, seconds: float):
        self.step(int(seconds / self.m.opt.timestep))

    # ---------------------------------------------------------------- kinematics
    def tcp_pose(self, data: mujoco.MjData | None = None) -> tuple[np.ndarray, np.ndarray]:
        data = self.d if data is None else data
        rot = data.xmat[self.hand_id].reshape(3, 3)
        return data.xpos[self.hand_id] + rot[:, 2] * TCP_OFFSET, rot.copy()

    def ik(self, pos: np.ndarray, rot: np.ndarray, q0: np.ndarray, iters: int = 60, tol: float = 5e-4) -> np.ndarray:
        d = self.ik_data
        d.qpos[:] = self.d.qpos
        q = q0.copy()
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            d.qpos[:7] = q
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            p, r = self.tcp_pose(d)
            err_p = pos - p
            err_r = Rotation.from_matrix(rot @ r.T).as_rotvec()
            if np.linalg.norm(err_p) < tol and np.linalg.norm(err_r) < 5 * tol:
                break
            mujoco.mj_jac(self.m, d, jacp, jacr, p, self.hand_id)
            jac = np.vstack([jacp[:, :7], jacr[:, :7]])
            err = np.concatenate([err_p, 0.6 * err_r])
            lam = 1e-3
            dq = jac.T @ np.linalg.solve(jac @ jac.T + lam * np.eye(6), err)
            # nullspace pull towards a comfortable posture keeps the elbow up
            null = np.eye(7) - np.linalg.pinv(jac) @ jac
            dq += null @ (0.05 * (HOME_Q - q))
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def nearest_yaw(self, yaw: float, symmetry: float) -> float:
        """Among yaw + k*symmetry pick the one closest to the current gripper yaw (least wrist motion)."""
        _, r = self.tcp_pose()
        cur = np.arctan2(r[1, 1], r[0, 1])
        k = np.round((cur - yaw) / symmetry)
        cand = yaw + k * symmetry
        if symmetry >= np.pi * 1.99:
            return cur
        return float(cand)

    # ---------------------------------------------------------------- motions
    def move_joints(self, q_target: np.ndarray, duration: float):
        q0 = self.q_des.copy()
        n = max(1, int(duration / (self.ctrl_steps * self.m.opt.timestep)))
        for i in range(1, n + 1):
            self.set_arm(q0 + (q_target - q0) * min_jerk(i / n))
            self.step(self.ctrl_steps)

    def move_to(self, pos, yaw: float, duration: float):
        p0, r0 = self.tcp_pose()
        p1, r1 = np.asarray(pos, float), down_rotation(yaw)
        slerp = Slerp([0, 1], Rotation.from_matrix([r0, r1]))
        q = self.q_des.copy()
        n = max(1, int(duration / (self.ctrl_steps * self.m.opt.timestep)))
        for i in range(1, n + 1):
            s = min_jerk(i / n)
            q = self.ik(p0 + (p1 - p0) * s, slerp(s).as_matrix(), q, iters=15)
            self.set_arm(q)
            self.step(self.ctrl_steps)
        self.set_arm(self.ik(p1, r1, q))
        self.wait(0.15)

    def gripper(self, value: float, duration: float = 0.5):
        self.d.ctrl[7] = value
        self.wait(duration)

    def finger_width(self) -> float:
        return float(self.d.qpos[7] + self.d.qpos[8])
