"""MuJoCo scene: Franka Panda on a table, colored objects and sorting trays."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "franka_emika_panda"

COLORS = {
    "red": (0.86, 0.12, 0.12),
    "blue": (0.12, 0.32, 0.92),
    "green": (0.10, 0.70, 0.22),
    "yellow": (0.98, 0.82, 0.08),
}
SHAPES = ("cube", "ball", "cylinder")

TRAY_HALF = 0.075
TRAY_WALL_H = 0.03

# Overhead perception camera (looking straight down, robot base at the bottom of the image).
OVERHEAD_CAM = dict(pos=(0.45, 0.0, 1.05), fovy=52.0)


@dataclasses.dataclass
class ObjectSpec:
    name: str
    color: str
    shape: str
    pos: tuple[float, float]
    yaw: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.color} {self.shape}"


@dataclasses.dataclass
class TraySpec:
    name: str
    color: str
    pos: tuple[float, float]

    @property
    def label(self) -> str:
        return f"{self.color} tray"


DEFAULT_TRAYS = [
    TraySpec("tray_red", "red", (0.30, 0.43)),
    TraySpec("tray_blue", "blue", (0.55, 0.43)),
    TraySpec("tray_green", "green", (0.42, -0.43)),
]


def _rgba(color: str) -> str:
    r, g, b = COLORS[color]
    return f"{r} {g} {b} 1"


def _object_xml(o: ObjectSpec) -> str:
    rgba = _rgba(o.color)
    common = f'rgba="{rgba}" friction="1.5 0.02 0.002" condim="4" solimp="0.95 0.99 0.001" solref="0.004 1"'
    if o.shape == "cube":
        geom = f'<geom type="box" size="0.022 0.022 0.022" mass="0.06" {common}/>'
        z = 0.022
    elif o.shape == "ball":
        geom = f'<geom type="sphere" size="0.024" mass="0.05" {common}/>'
        z = 0.024
    elif o.shape == "cylinder":
        # MuJoCo's cylinder-vs-box contacts are unstable against the small fingertip pads, so the
        # collision shape is a 24-sided prism mesh; the smooth cylinder is only drawn.
        geom = (f'<geom type="cylinder" size="0.021 0.03" rgba="{rgba}" contype="0" conaffinity="0" mass="0"/>'
                f'<geom type="mesh" mesh="cyl_prism" mass="0.06" group="3" {common}/>')
        z = 0.03
    else:
        raise ValueError(o.shape)
    half = o.yaw / 2.0
    quat = f"{np.cos(half):.6f} 0 0 {np.sin(half):.6f}"
    return (
        f'<body name="{o.name}" pos="{o.pos[0]:.4f} {o.pos[1]:.4f} {z + 0.001:.4f}" quat="{quat}">'
        f'<freejoint name="{o.name}_joint"/>{geom}</body>'
    )


def _tray_xml(t: TraySpec) -> str:
    rgba = _rgba(t.color)
    h, wh, th = TRAY_HALF, TRAY_WALL_H / 2, 0.004
    x, y = t.pos
    parts = [f'<geom type="box" size="{h} {h} {th / 2}" pos="0 0 {th / 2}" rgba="{rgba}"/>']
    for sx, sy, ex, ey in ((h, 0, th, h), (-h, 0, th, h), (0, h, h, th), (0, -h, h, th)):
        parts.append(f'<geom type="box" size="{ex} {ey} {wh}" pos="{sx} {sy} {wh}" rgba="{rgba}"/>')
    return f'<body name="{t.name}" pos="{x} {y} 0">{"".join(parts)}</body>'


def build_scene_xml(objects: list[ObjectSpec], trays: list[TraySpec]) -> str:
    cam = OVERHEAD_CAM
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    prism = " ".join(f"{0.021 * np.cos(a):.5f} {0.021 * np.sin(a):.5f} {zz}" for zz in (-0.03, 0.03) for a in ang)
    objs = "\n    ".join(_object_xml(o) for o in objects)
    trs = "\n    ".join(_tray_xml(t) for t in trays)
    return f"""<mujoco model="pai tidy-up scene">
  <include file="panda.xml"/>
  <option timestep="0.002" impratio="10" cone="elliptic">
    <flag multiccd="enable"/>
  </option>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight diffuse="0.35 0.35 0.35" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <mesh name="cyl_prism" vertex="{prism}"/>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.65 0.78" rgb2="0.1 0.12 0.16" width="512" height="3072"/>
    <texture type="2d" name="wood" builtin="flat" rgb1="0.78 0.70 0.58" width="64" height="64"/>
    <material name="table" texture="wood" rgba="0.62 0.56 0.48 1" reflectance="0.02"/>
    <texture type="2d" name="floor" builtin="checker" rgb1="0.32 0.34 0.38" rgb2="0.26 0.28 0.32" width="300" height="300"/>
    <material name="floor" texture="floor" texrepeat="8 8" texuniform="true"/>
  </asset>
  <worldbody>
    <light pos="0.6 -0.6 2.5" dir="-0.1 0.25 -1" directional="true" castshadow="true" diffuse="0.45 0.45 0.45"/>
    <light pos="1.5 1.0 1.8" dir="-0.6 -0.4 -1" directional="true" castshadow="false" diffuse="0.25 0.25 0.25"/>
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 -0.75" material="floor"/>
    <geom name="table" type="box" size="0.65 0.75 0.375" pos="0.35 0 -0.375" material="table"/>
    <camera name="overhead" pos="{cam['pos'][0]} {cam['pos'][1]} {cam['pos'][2]}" xyaxes="0 -1 0 1 0 0" fovy="{cam['fovy']}"/>
    {trs}
    {objs}
  </worldbody>
</mujoco>
"""


def random_layout(rng: np.random.Generator, combos: list[tuple[str, str]], min_dist: float = 0.085) -> list[ObjectSpec]:
    """Scatter objects in the workspace in front of the robot without overlaps."""
    placed: list[ObjectSpec] = []
    for i, (color, shape) in enumerate(combos):
        for _ in range(1000):
            p = (rng.uniform(0.36, 0.66), rng.uniform(-0.24, 0.24))
            if all(np.hypot(p[0] - q.pos[0], p[1] - q.pos[1]) > min_dist for q in placed):
                break
        placed.append(ObjectSpec(f"obj{i}_{color}_{shape}", color, shape, p, float(rng.uniform(-np.pi / 4, np.pi / 4))))
    return placed


def write_scene(objects: list[ObjectSpec], trays: list[TraySpec], filename: str = "_generated_scene.xml") -> Path:
    path = ASSET_DIR / filename
    path.write_text(build_scene_xml(objects, trays))
    return path
