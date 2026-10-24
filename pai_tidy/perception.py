"""Perception: open-vocabulary detection (OWLv2) + depth-based 3D geometric verification."""

from __future__ import annotations

import colorsys
import dataclasses

import mujoco
import numpy as np
from scipy import ndimage

from .scene import COLORS, SHAPES

OWL_MODEL_ID = "google/owlv2-base-patch16-ensemble"
TABLE_Z = 0.0


@dataclasses.dataclass
class Detection:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    score: float  # OWLv2 confidence of the best text query
    owl_label: str  # raw OWLv2 label (before verification)
    kind: str  # "object" | "tray"
    color: str
    shape: str  # cube | ball | cylinder | tray
    position: np.ndarray  # grasp/center point in world frame (x, y, z)
    height: float
    footprint: float  # longest side of the xy footprint (m)
    yaw: float  # in-plane orientation (rad), meaningful for cubes / trays

    @property
    def label(self) -> str:
        return f"{self.color} {self.shape}" if self.kind == "object" else f"{self.color} tray"


class Camera:
    """Renders RGB + metric depth from a named MuJoCo camera and back-projects pixels to 3D."""

    def __init__(self, model: mujoco.MjModel, name: str, width: int = 960, height: int = 720):
        self.model, self.name, self.w, self.h = model, name, width, height
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        self.renderer = mujoco.Renderer(model, height, width)
        fovy = np.deg2rad(model.cam_fovy[self.cam_id])
        self.f = 0.5 * height / np.tan(fovy / 2)

    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.renderer.update_scene(data, camera=self.name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.name)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth, self.backproject(data, depth)

    def close(self):
        self.renderer.close()

    def backproject(self, data: mujoco.MjData, depth: np.ndarray) -> np.ndarray:
        """Returns an (H, W, 3) array of world coordinates."""
        u, v = np.meshgrid(np.arange(self.w) + 0.5, np.arange(self.h) + 0.5)
        x = (u - self.w / 2) / self.f * depth
        y = -(v - self.h / 2) / self.f * depth
        pts_cam = np.stack([x, y, -depth], axis=-1)
        rot = data.cam_xmat[self.cam_id].reshape(3, 3)
        return pts_cam @ rot.T + data.cam_xpos[self.cam_id]


def _hue_color(rgb_pixels: np.ndarray) -> tuple[str, float]:
    """Nearest reference color in hue space, ignoring dark (shadowed) pixels."""
    px = rgb_pixels.reshape(-1, 3) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in px[:: max(1, len(px) // 400)]])
    hsv = hsv[(hsv[:, 1] > 0.35) & (hsv[:, 2] > 0.25)]
    if len(hsv) == 0:
        return "unknown", 0.0
    # dominant hue (histogram mode) so that items lying inside a tray do not skew the tray colour
    hist, edges = np.histogram(hsv[:, 0], bins=36, range=(0, 1))
    mean_h = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
    best, best_d = "unknown", 1.0
    for name, ref in COLORS.items():
        h = colorsys.rgb_to_hsv(*ref)[0]
        dist = min(abs(mean_h - h), 1 - abs(mean_h - h))
        if dist < best_d:
            best, best_d = name, dist
    return best, 1.0 - best_d * 4


def _min_area_rect(xy: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Brute-force rotating calipers: (area, yaw in [0, pi/2), extents) of the min-area bounding box."""
    c = xy - xy.mean(axis=0)
    best = (np.inf, 0.0, np.zeros(2))
    for th in np.deg2rad(np.arange(0, 90, 2.0)):
        r = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        p = c @ r.T
        ext = p.max(axis=0) - p.min(axis=0)
        if ext[0] * ext[1] < best[0]:
            best = (ext[0] * ext[1], th, ext)
    return best


def _min_area_yaw(xy: np.ndarray) -> tuple[float, float]:
    _, yaw, ext = _min_area_rect(xy)
    return yaw, float(ext.max())


def classify_shape(height: float, top_flat_ratio: float, footprint: float) -> str:
    if footprint > 0.10:
        return "tray"
    if top_flat_ratio < 0.6:  # a sphere has only a small cap near its highest point
        return "ball"
    return "cylinder" if height > 0.054 else "cube"


class Perception:
    def __init__(self, device: str = "cpu", score_threshold: float = 0.08):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.torch = torch
        self.processor = Owlv2Processor.from_pretrained(OWL_MODEL_ID)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL_ID).to(device).eval()
        self.device = device
        self.threshold = score_threshold
        self.queries = [f"a {c} {s}" for c in COLORS for s in SHAPES] + [f"a {c} tray" for c in COLORS]

    def owl_detect(self, rgb: np.ndarray) -> list[tuple[np.ndarray, float, str]]:
        from PIL import Image

        img = Image.fromarray(rgb)
        inputs = self.processor(text=[self.queries], images=img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        side = max(img.size)  # OWLv2 pads to a square, so boxes are in padded-square coordinates
        res = self.processor.post_process_grounded_object_detection(
            out, threshold=self.threshold, target_sizes=self.torch.tensor([[side, side]])
        )[0]
        dets = [
            (np.array(b), float(s), self.queries[int(l)][2:])
            for b, s, l in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist())
        ]
        return _nms(dets, iou_thr=0.4)

    def detect(self, rgb: np.ndarray, xyz: np.ndarray, owl_dets=None) -> list[Detection]:
        h, w = xyz.shape[:2]
        z = xyz[..., 2]
        results: list[Detection] = []
        for box, score, owl_label in owl_dets if owl_dets is not None else self.owl_detect(rgb):
            x0, y0, x1, y1 = np.clip(box, 0, [w, h, w, h]).astype(int)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            zb = z[y0:y1, x0:x1]
            # 1) is the box a tray? (large connected region slightly above the table)
            comp = _largest_component(zb > TABLE_Z + 0.0025)
            if comp is None:
                continue
            pts = xyz[y0:y1, x0:x1][comp]
            if pts[:, 2].max() > 0.2:  # robot links, not a tabletop item
                continue
            yaw, footprint = _min_area_yaw(pts[:, :2])
            if footprint > 0.10:
                lo, hi = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
                center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])
                color = _hue_color(rgb[y0:y1, x0:x1][comp])[0]
                results.append(Detection((x0, y0, x1, y1), score, owl_label, "tray", color, "tray",
                                         center, float(pts[:, 2].max()), footprint, yaw))
                continue
            # 2) object: segment relative to the local support surface (table or tray floor)
            support = float(np.percentile(zb, 10))
            comp = _largest_component(zb > support + 0.008)
            if comp is None:
                continue
            pts = xyz[y0:y1, x0:x1][comp]
            yaw, footprint = _min_area_yaw(pts[:, :2])
            top_z = float(np.percentile(pts[:, 2], 97))
            height = top_z - support
            top = pts[:, 2] > top_z - 0.006
            if top.sum() > 10:
                yaw = _min_area_rect(pts[top, :2])[1]  # orientation from the top face only
            shape = classify_shape(height, float(top.mean()), footprint)
            if shape == "tray":
                continue
            color = _hue_color(rgb[y0:y1, x0:x1][comp])[0]
            center = np.array([*pts[top, :2].mean(axis=0), top_z])
            results.append(Detection((x0, y0, x1, y1), score, owl_label, "object", color, shape,
                                     center, height, footprint, yaw))
        return _dedupe_3d(results)


def _largest_component(mask: np.ndarray, min_px: int = 20) -> np.ndarray | None:
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    comp = lab == (int(np.argmax(sizes)) + 1)
    return comp if comp.sum() >= min_px else None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _nms(dets, iou_thr):
    dets = sorted(dets, key=lambda d: -d[1])
    keep = []
    for d in dets:
        if all(_iou(d[0], k[0]) < iou_thr for k in keep):
            keep.append(d)
    return keep


def _dedupe_3d(dets: list[Detection]) -> list[Detection]:
    """Two boxes that back-project to the same physical item are merged (keep the higher score)."""
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        radius = 0.06 if d.kind == "tray" else 0.025
        if all(np.linalg.norm(d.position[:2] - k.position[:2]) > radius or d.kind != k.kind for k in keep):
            keep.append(d)
    return keep
