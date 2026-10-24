"""Video recorder: main 3D view + overlays (instruction, plan progress, AI perception picture-in-picture)."""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path("/usr/share/fonts/opentype/noto")
W, H = 1280, 720
TOP, BOTTOM, SIDE = 50, 48, 400  # top bar, status bar, right side panel
VIEW_W, VIEW_H = W - SIDE, H - TOP - BOTTOM

STATUS_COLORS = {"pending": (150, 150, 150), "active": (255, 196, 0), "done": (60, 200, 90),
                 "skipped": (230, 90, 70)}
LABEL_RGB = {"red": (235, 60, 60), "blue": (70, 130, 255), "green": (40, 200, 80), "yellow": (250, 210, 30),
             "unknown": (200, 200, 200)}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / ("NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc")
    try:
        return ImageFont.truetype(str(path), size, index=0)
    except OSError:
        return ImageFont.load_default(size)


def annotate_detections(rgb: np.ndarray, dets, verbose: bool = True) -> Image.Image:
    """Draw OWLv2 boxes; orange label = the raw OWLv2 label was corrected by 3D verification."""
    img = Image.fromarray(rgb).convert("RGB")
    dr = ImageDraw.Draw(img)
    f = font(17 if verbose else 36, bold=True)
    for d in dets:
        x0, y0, x1, y1 = d.box
        corrected = d.owl_label != d.label
        col = (255, 150, 0) if corrected else (40, 230, 120)
        dr.rectangle([x0, y0, x1, y1], outline=col, width=3 if verbose else 5)
        text = d.label + (f"  (OWL: {d.owl_label})" if corrected and verbose else "")
        if not verbose and d.kind == "tray":
            continue
        tw = dr.textlength(text, font=f)
        th = 22 if verbose else 44
        ty = y0 - th - 2 if y0 > th + 4 else y1 + 2
        dr.rectangle([x0, ty, x0 + tw + 8, ty + th], fill=(0, 0, 0))
        dr.text((x0 + 4, ty), text, font=f, fill=col)
    return img


class Recorder:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, path: str | Path, fps: int = 25):
        self.m, self.d, self.fps = model, data, fps
        self.renderer = mujoco.Renderer(model, VIEW_H, VIEW_W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.44, 0.0, 0.0]
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 1.5, -40.0, 180.0
        self.writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                                         macro_block_size=8, ffmpeg_log_level="error")
        self.title = ""
        self.instruction = ""
        self.parser_name = ""
        self.plan: list[tuple[str, str]] = []
        self.status = ""
        self.banner = ""
        self.banner_frames = 0
        self.pip: Image.Image | None = None
        self.n_frames = 0
        self.f_title, self.f_mid, self.f_small = font(26, True), font(22), font(18)

    # ------------------------------------------------------------ writing
    def _write(self, img: Image.Image):
        self.writer.append_data(np.asarray(img.convert("RGB")))
        self.n_frames += 1

    @property
    def seconds(self) -> float:
        return self.n_frames / self.fps

    def frame(self):
        """Called by the simulation at the video frame rate."""
        t = self.d.time
        self.cam.azimuth = 180.0 + 14.0 * math.sin(t / 9.0)
        self.cam.elevation = -40.0 + 4.0 * math.sin(t / 13.0)
        self.renderer.update_scene(self.d, camera=self.cam)
        img = Image.new("RGB", (W, H), (16, 20, 30))
        img.paste(Image.fromarray(self.renderer.render()), (0, TOP))
        self._write(self._overlay(img))

    def hold_panel(self, panel: Image.Image, seconds: float):
        img = self._overlay(panel.resize((W, H)), minimal=True)
        for _ in range(int(seconds * self.fps)):
            self._write(img)

    def card(self, heading: str, lines: list[str], seconds: float, sub: str = ""):
        img = Image.new("RGB", (W, H), (18, 22, 32))
        dr = ImageDraw.Draw(img)
        for y in range(H):
            c = int(18 + 20 * y / H)
            dr.line([(0, y), (W, y)], fill=(c, c + 4, c + 16))
        dr.text((80, 90), heading, font=font(46, True), fill=(255, 255, 255))
        if sub:
            dr.text((82, 160), sub, font=font(24), fill=(150, 200, 255))
        y = 230
        for line in lines:
            dr.text((90, y), line, font=font(27), fill=(225, 230, 240))
            y += 50
        for _ in range(int(seconds * self.fps)):
            self._write(img)

    def show_banner(self, text: str, seconds: float = 2.5):
        self.banner, self.banner_frames = text, int(seconds * self.fps)

    def close(self):
        self.writer.close()
        self.renderer.close()

    # ------------------------------------------------------------ overlays
    def _overlay(self, img: Image.Image, minimal: bool = False) -> Image.Image:
        img = img.convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)
        # top bar
        dr.rectangle([0, 0, W, 50], fill=(10, 14, 24, 215))
        dr.text((18, 9), self.title, font=self.f_title, fill=(255, 255, 255))
        speed = "等倍速 (1x)"
        dr.text((W - dr.textlength(speed, font=self.f_mid) - 18, 12), speed, font=self.f_mid, fill=(255, 210, 90))
        if not minimal:
            self._side_panel(dr, layer)
        # status bar
        if self.status:
            dr.rectangle([0, H - 48, W, H], fill=(10, 14, 24, 215))
            dr.text((18, H - 41), self.status, font=self.f_mid, fill=(255, 255, 255))
        # banner (events such as failures / disturbances)
        if self.banner_frames > 0:
            self.banner_frames -= 1
            f = font(30, True)
            tw = dr.textlength(self.banner, font=f)
            bx = ((W - SIDE) - tw) / 2 if not minimal else (W - tw) / 2
            dr.rounded_rectangle([bx - 20, 80, bx + tw + 20, 132], radius=10, fill=(200, 50, 40, 235))
            dr.text((bx, 85), self.banner, font=f, fill=(255, 255, 255))
        return Image.alpha_composite(img, layer)

    def _side_panel(self, dr: ImageDraw.ImageDraw, layer: Image.Image):
        x0, x1 = W - SIDE, W
        dr.rectangle([x0, TOP, x1, H - BOTTOM], fill=(16, 20, 30, 255))
        x = x0 + 14
        y = TOP + 10
        if self.instruction:
            dr.text((x, y), "言語指示", font=self.f_small, fill=(170, 200, 255))
            y += 26
            lines = _wrap(dr, f"「{self.instruction}」", self.f_small, SIDE - 36)
            dr.rounded_rectangle([x - 4, y - 2, x1 - 10, y + 26 * len(lines) + 6], radius=6, fill=(250, 250, 252, 255))
            for line in lines:
                dr.text((x + 4, y + 2), line, font=self.f_small, fill=(20, 20, 30))
                y += 26
            y += 16
        if self.plan:
            dr.text((x, y), f"実行計画（{self.parser_name}）", font=self.f_small, fill=(170, 200, 255))
            y += 28
            for text, st in self.plan:
                dr.ellipse([x + 2, y + 5, x + 16, y + 19], fill=STATUS_COLORS[st])
                dr.text((x + 26, y), text, font=self.f_small,
                        fill=(240, 240, 240) if st != "pending" else (150, 150, 150))
                y += 28
            y += 10
        if self.pip is not None:
            pw = SIDE - 24
            ph = pw * self.pip.height // self.pip.width
            py = H - BOTTOM - ph - 12
            dr.text((x, py - 26), "AI知覚: OWLv2 + 深度による3D検証", font=self.f_small, fill=(170, 200, 255))
            layer.paste(self.pip.resize((pw, ph)), (x - 2, py))


def _wrap(dr: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        if dr.textlength(cur + ch, font=f) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    return lines + [cur] if cur else lines


def perception_panel(raw_rgb: np.ndarray, annotated: Image.Image, dets, infer_s: float) -> Image.Image:
    """Full-screen explanation of one observation: raw AI output vs. verified result."""
    img = Image.new("RGB", (W, H), (16, 20, 30))
    dr = ImageDraw.Draw(img)
    view = annotated.resize((800, 600))
    img.paste(view, (20, 70))
    dr.text((24, 676), "緑枠 = OWLv2のラベルが正しい / 橙枠 = 深度の3D形状・色で補正", font=font(17), fill=(200, 210, 230))
    x = 840
    dr.text((x, 84), "AIによる物体認識", font=font(28, True), fill=(255, 255, 255))
    dr.text((x, 124), f"OWLv2 (open-vocabulary) 推論 {infer_s:.1f}s / CPU", font=font(17), fill=(150, 200, 255))
    y = 165
    fs, fb = font(17), font(17, True)
    dr.text((x, y), "OWLv2の出力", font=fb, fill=(180, 180, 190))
    dr.text((x + 215, y), "→ 3D検証後", font=fb, fill=(180, 180, 190))
    y += 28
    for d in sorted(dets, key=lambda d: (d.kind, d.label)):
        corrected = d.owl_label != d.label
        dr.text((x, y), d.owl_label, font=fs, fill=(255, 150, 0) if corrected else (200, 200, 200))
        dr.text((x + 215, y), d.label, font=fb, fill=LABEL_RGB.get(d.color, (255, 255, 255)))
        y += 26
    n_fix = sum(d.owl_label != d.label for d in dets)
    y += 12
    dr.text((x, y), f"検出 {len(dets)} 件 / 補正 {n_fix} 件", font=font(20, True), fill=(255, 210, 90))
    y += 40
    for line in ("深度画像 → 3D点群へ逆投影", "高さ・上面の平坦さ・外形で形状判定", "色相ヒストグラムで色を判定",
                 "最小外接矩形でキューブの向き推定"):
        dr.text((x, y), "・" + line, font=fs, fill=(210, 215, 225))
        y += 26
    return img
