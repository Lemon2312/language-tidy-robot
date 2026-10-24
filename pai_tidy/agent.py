"""Closed-loop tidy-up agent: observe -> select -> pick -> place -> verify (re-observe) -> retry."""

from __future__ import annotations

import dataclasses
import time
from typing import Callable

import mujoco
import numpy as np

from .language import SAME, Step
from .perception import Camera, Detection, Perception
from .robot import GRIPPER_CLOSED, GRIPPER_OPEN, GRIPPER_RELEASE, HOME_Q, OBSERVE_Q, Robot
from .scene import TRAY_HALF

TRAY_FLOOR = 0.004
from .video import Recorder, annotate_detections, perception_panel

JA_COLOR = {"red": "赤", "blue": "青", "green": "緑", "yellow": "黄"}
JA_SHAPE = {"cube": "キューブ", "ball": "ボール", "cylinder": "円柱"}
SLOTS = [(-0.027, -0.027), (0.027, 0.027), (-0.027, 0.027), (0.027, -0.027)]


def ja(det: Detection) -> str:
    return f"{JA_COLOR.get(det.color, det.color)}の{JA_SHAPE.get(det.shape, det.shape)}"


@dataclasses.dataclass
class Stats:
    picks: int = 0
    placed: int = 0
    grasp_failures: int = 0
    retries: int = 0
    skipped: list[str] = dataclasses.field(default_factory=list)
    owl_boxes: int = 0
    owl_corrected: int = 0
    infer_seconds: list[float] = dataclasses.field(default_factory=list)


class TidyAgent:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot: Robot, camera: Camera,
                 perception: Perception, rec: Recorder | None = None, log: Callable[[str], None] = print):
        self.m, self.d, self.robot, self.camera, self.perception = model, data, robot, camera, perception
        self.rec, self.log = rec, log
        self.stats = Stats()
        self.fault_offsets: dict[str, float] = {}  # label -> grasp offset (m) injected once (robustness test)
        self.after_place: dict[int, Callable[[], None]] = {}  # n-th placement -> disturbance hook
        self.at_observe = False
        self.cached: list[Detection] | None = None

    # ------------------------------------------------------------ helpers
    def say(self, status: str):
        self.log(status)
        if self.rec:
            self.rec.status = status

    def banner(self, text: str, seconds: float = 2.5):
        self.log(f"[event] {text}")
        if self.rec:
            self.rec.show_banner(text, seconds)

    def observe(self, explain: bool = False) -> list[Detection]:
        if self.cached is not None and not explain:
            return self.cached
        if not self.at_observe:
            self.say("観測姿勢へ移動（アームをカメラの視野から退避）")
            self.robot.move_joints(OBSERVE_Q, 1.1)
            self.at_observe = True
        self.robot.wait(0.25)
        rgb, _, xyz = self.camera.capture(self.d)
        crop = slice((rgb.shape[1] - 960) // 2, (rgb.shape[1] + 960) // 2) if rgb.shape[1] > 960 else slice(None)
        rgb, xyz = rgb[:, crop], xyz[:, crop]
        t0 = time.time()
        owl = self.perception.owl_detect(rgb)
        infer = time.time() - t0
        dets = self.perception.detect(rgb, xyz, owl_dets=owl)
        self.stats.infer_seconds.append(infer)
        self.stats.owl_boxes += len(dets)
        self.stats.owl_corrected += sum(d.owl_label != d.label for d in dets)
        self.log("  detections: " + ", ".join(f"{d.label}@({d.position[0]:.2f},{d.position[1]:.2f})" for d in dets))
        if self.rec:
            self.rec.pip = annotate_detections(rgb, dets, verbose=False).crop((60, 90, 900, 600))  # tabletop only
            if explain:
                ann = annotate_detections(rgb, dets)
                self.rec.hold_panel(perception_panel(rgb, ann, dets, infer), 6.0)
        self.cached = dets
        return dets

    @staticmethod
    def in_tray(obj: Detection, tray: Detection) -> bool:
        return bool(np.all(np.abs(obj.position[:2] - tray.position[:2]) < TRAY_HALF + 0.005))

    def free_slot(self, tray: Detection, objects: list[Detection]) -> tuple[np.ndarray, float]:
        inside = [o.position[:2] for o in objects if self.in_tray(o, tray)]
        best, best_d = None, -1.0
        for sx, sy in SLOTS:
            p = tray.position[:2] + [sx, sy]
            dmin = min((np.linalg.norm(p - q) for q in inside), default=1.0)
            if dmin > best_d + 1e-6:
                best, best_d = p, dmin
        return best, best_d

    # ------------------------------------------------------------ main loop
    def run(self, steps: list[Step], plan_view: list | None = None, max_actions: int = 20) -> Stats:
        first = True
        actions = 0
        for si, step in enumerate(steps):
            if plan_view is not None:
                plan_view[si] = (plan_view[si][0], "active")
            skipped_here = False
            while actions < max_actions:
                dets = self.observe(explain=first)
                first = False
                trays = {d.color: d for d in dets if d.kind == "tray"}
                objects = [d for d in dets if d.kind == "object"]
                loose = [o for o in objects if not any(self.in_tray(o, t) for t in trays.values())]
                cands = [o for o in loose if step.target.matches(o.color, o.shape)]
                if not cands:
                    break
                todo = []
                for o in cands:
                    dest = o.color if step.dest == SAME else step.dest
                    if dest in trays:
                        todo.append((o, trays[dest]))
                    elif f"{o.label}" not in self.stats.skipped:
                        self.stats.skipped.append(o.label)
                        skipped_here = True
                        self.banner(f"{ja(o)}: {JA_COLOR.get(dest, dest)}色のトレイが無いため保留", 3.0)
                        self.say(f"判断: {ja(o)} の行き先（{JA_COLOR.get(dest, dest)}のトレイ）が見つからない → 動かさない")
                        self.robot.wait(1.5)
                if not todo:
                    break
                # nearest item first (shortest reach, least risk of sweeping other items)
                obj, tray = min(todo, key=lambda ot: np.linalg.norm(ot[0].position[:2] - [0.3, 0.0]))
                target, clearance = self.free_slot(tray, objects)
                actions += 1
                ok = self.pick_and_place(obj, tray, target, stack=clearance < 0.045)
                if ok:
                    self.stats.placed += 1
                    hook = self.after_place.pop(self.stats.placed, None)
                    if hook:
                        hook()
            if plan_view is not None:
                plan_view[si] = (plan_view[si][0], "skipped" if skipped_here else "done")
        self.say("すべての指示を完了しました")
        self.robot.move_joints(OBSERVE_Q, 1.0)
        self.at_observe = True
        return self.stats

    def pick_and_place(self, obj: Detection, tray: Detection, target_xy: np.ndarray, stack: bool = False) -> bool:
        R = self.robot
        self.stats.picks += 1
        self.at_observe = False
        self.cached = None
        x, y, top = obj.position
        name = ja(obj)
        tray_name = f"{JA_COLOR[tray.color]}のトレイ"
        self.say(f"把持へ: {name}  (x={obj.position[0]:.3f}, y={obj.position[1]:.3f} m) → {tray_name}")
        R.move_joints(HOME_Q, 0.9)
        sym = np.pi / 2 if obj.shape == "cube" else 2 * np.pi
        yaw = R.nearest_yaw(obj.yaw, sym)
        offset = self.fault_offsets.pop(obj.label, None)
        if offset is not None:  # shift perpendicular to the closing direction so the fingers miss
            x, y = x - offset * np.sin(yaw), y + offset * np.cos(yaw)
        bottom = top - obj.height
        grasp_z = max(bottom + obj.height / 2, 0.02)  # mid-height grasp keeps tall items from tipping
        R.gripper(GRIPPER_OPEN, 0.0)
        R.move_to([x, y, 0.15], yaw, 0.9)
        R.move_to([x, y, grasp_z], yaw, 0.7)
        R.gripper(GRIPPER_CLOSED, 0.6)
        if R.finger_width() < 0.012:
            self.stats.grasp_failures += 1
            self.stats.retries += 1
            self.banner("把持失敗を検知（指の開き幅≈0）→ 再観測してリトライ", 3.0)
            self.say(f"把持失敗: {name}（グリッパが空）→ 再観測")
            R.gripper(GRIPPER_OPEN, 0.3)
            R.move_to([x, y, 0.15], yaw, 0.6)
            return False
        self.say(f"運搬中: {name} → {tray_name}")
        R.move_to([x, y, 0.18], yaw, 0.6)
        tx, ty = target_xy
        R.move_to([tx, ty, 0.18], yaw, 1.0)
        if R.finger_width() < 0.012:
            self.stats.retries += 1
            self.banner("運搬中の落下を検知 → 再観測してリトライ", 3.0)
            return False
        # lower until the item is ~1 cm above the tray floor, then open only partially so the
        # fingers cannot hit the tray walls
        release_z = TRAY_FLOOR + (grasp_z - bottom) + 0.01 + (0.055 if stack else 0.0)
        R.move_to([tx, ty, release_z + 0.04], yaw, 0.4)
        R.move_to([tx, ty, release_z], yaw, 0.35)
        R.gripper(GRIPPER_RELEASE, 0.35)
        R.move_to([tx, ty, 0.18], yaw, 0.45)
        R.gripper(GRIPPER_OPEN, 0.0)
        self.say(f"配置完了: {name} → {tray_name}（次の観測で結果を検証）")
        return True


def move_object(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, xy: tuple[float, float]):
    """Teleport a free object (used to simulate a person moving things around mid-task)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    adr = model.jnt_qposadr[model.body_jntadr[bid]]
    data.qpos[adr:adr + 2] = xy
    data.qvel[model.jnt_dofadr[model.body_jntadr[bid]]:][:6] = 0
    mujoco.mj_forward(model, data)
