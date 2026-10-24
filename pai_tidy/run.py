"""Entry point.

    python -m pai_tidy.run --demo                       # records the submission video (2 scenarios)
    python -m pai_tidy.run -i "赤いものを青いトレイへ" --seed 3   # any instruction on a random scene
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .agent import TidyAgent, move_object  # noqa: E402
from .language import SAME, Step, make_parser  # noqa: E402
from .perception import Camera, Perception  # noqa: E402
from .robot import Robot  # noqa: E402
from .scene import DEFAULT_TRAYS, TRAY_HALF, ObjectSpec, random_layout, write_scene  # noqa: E402
from .video import Recorder  # noqa: E402

ALL_COMBOS = [(c, s) for c in ("red", "blue", "green", "yellow") for s in ("cube", "ball", "cylinder")]


def expected_tray(obj: ObjectSpec, steps: list[Step]) -> str | None:
    for st in steps:
        if st.target.matches(obj.color, obj.shape):
            dest = obj.color if st.dest == SAME else st.dest
            return dest if any(t.color == dest for t in DEFAULT_TRAYS) else None
    return None


def evaluate(model, data, objects: list[ObjectSpec], steps: list[Step]) -> dict:
    """Ground-truth check from the simulator state (not used by the agent itself)."""
    rows, ok = [], 0
    for o in objects:
        pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, o.name)]
        actual = next((t.color for t in DEFAULT_TRAYS
                       if abs(pos[0] - t.pos[0]) < TRAY_HALF and abs(pos[1] - t.pos[1]) < TRAY_HALF), None)
        exp = expected_tray(o, steps)
        good = actual == exp
        ok += good
        rows.append({"object": o.label, "expected": exp or "stay", "actual": actual or "table", "ok": good})
    return {"success": ok, "total": len(objects), "rows": rows}


class Session:
    def __init__(self, objects: list[ObjectSpec], perception: Perception, video: Path | None, fps: int = 25):
        path = write_scene(objects, DEFAULT_TRAYS)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.objects = objects
        self.rec = Recorder(self.model, self.data, video, fps) if video else None
        self.robot = Robot(self.model, self.data, fps=fps, on_frame=self.rec.frame if self.rec else None)
        self.robot.reset()
        mujoco.mj_forward(self.model, self.data)
        self.camera = Camera(self.model, "overhead", 1280, 720)
        self.agent = TidyAgent(self.model, self.data, self.robot, self.camera, perception, self.rec)

    def close(self):
        # OSMesa misrenders when stale contexts from a previous model are still alive
        self.camera.close()
        if self.rec:
            self.rec.close()


def run_scenario(sess: Session, title: str, instruction: str, parser) -> dict:
    steps = parser.parse(instruction)
    print(f"\n=== {title}\ninstruction: {instruction}")
    for s in steps:
        print("  plan:", s.describe_ja())
    plan_view = [(s.describe_ja(), "pending") for s in steps]
    if sess.rec:
        sess.rec.title, sess.rec.instruction, sess.rec.parser_name = title, instruction, parser.name
        sess.rec.plan = plan_view
        sess.rec.status = "言語指示を解析 → 実行計画を生成"
    sess.robot.wait(1.5)
    t0 = time.time()
    stats = sess.agent.run(steps, plan_view)
    sess.robot.wait(1.5)
    result = evaluate(sess.model, sess.data, sess.objects, steps)
    result.update(title=title, instruction=instruction, plan=[s.describe_ja() for s in steps],
                  stats=stats.__dict__, wall_seconds=round(time.time() - t0, 1), sim_seconds=round(sess.data.time, 1))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def demo(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    perception = Perception()
    parser = make_parser()
    results = []
    video = out_dir / "demo.mp4"
    parts = [out_dir / "part1.mp4", out_dir / "part2.mp4"]

    # ---- Scenario 1: basic colour sorting
    objs1 = random_layout(np.random.default_rng(7), [("red", "cube"), ("blue", "ball"), ("green", "cylinder"),
                                                      ("blue", "cube")])
    s1 = Session(objs1, perception, parts[0])
    s1.rec.card("言語指示で動く 片付けロボット", [
        "・自然言語の指示 → 実行計画（言語理解）",
        "・OWLv2（オープン語彙物体検出AI）でカメラ画像から物体を発見",
        "・深度画像から3D形状を復元し、AIの誤認識を検証・補正",
        "・逆運動学 + 重力補償で Franka Panda を制御し把持・配置",
        "・毎回観測し直す閉ループ：失敗・外乱から自動復帰",
    ], 9.0, sub="Physical AI 応用1講座 最終課題 ／ MuJoCo シミュレーション（等倍速）")
    results.append(run_scenario(s1, "シナリオ1: 色ごとの片付け", "テーブルの上のものを同じ色のトレイに片付けて", parser))
    s1.close()

    # ---- Scenario 2: compound instruction + infeasible request + failure injection + human disturbance
    objs2 = random_layout(np.random.default_rng(11), [("red", "ball"), ("blue", "ball"), ("yellow", "cube"),
                                                       ("green", "cube"), ("red", "cylinder"), ("yellow", "cylinder")])
    s2 = Session(objs2, perception, parts[1])
    s2.rec.card("シナリオ2: 複合指示とロバスト性", [
        "・複数の条件を含む指示を順序付きの計画に分解",
        "・対応するトレイが無い物体は「動かさない」と判断",
        "・故障注入: 1回目の把持位置を意図的に4cmずらす → 失敗を検知して再試行",
        "・外乱: 作業中に人が物体を動かす → 再観測で追従",
    ], 7.0)
    s2.agent.fault_offsets["blue ball"] = 0.04
    moved = next(o for o in objs2 if o.label == "red cylinder")

    def disturbance():
        s2.agent.banner("外乱: 人が赤い円柱を別の場所へ動かしました", 3.0)
        move_object(s2.model, s2.data, moved.name, (0.62, -0.05))
        s2.robot.wait(1.0)

    s2.agent.after_place[1] = disturbance
    results.append(run_scenario(
        s2, "シナリオ2: 複合指示 + 失敗・外乱からの復帰",
        "ボールは全部青いトレイに入れて、黄色いキューブは緑の箱へ。それから残りを色ごとに片付けて", parser))
    total_ok = sum(r["success"] for r in results)
    total = sum(r["total"] for r in results)
    st = [r["stats"] for r in results]
    s2.rec.card("結果", [
        f"・最終配置の正解率: {total_ok}/{total} 物体（シミュレータの真値で判定）",
        f"・OWLv2のラベルを3D検証で補正した件数: {sum(s['owl_corrected'] for s in st)} / {sum(s['owl_boxes'] for s in st)} 検出",
        f"・把持失敗の検知と再試行: {sum(s['grasp_failures'] for s in st)} 回 → すべて復帰",
        "・トレイの無い黄色い円柱は指示どおり保留（テーブルに残す）",
        "・応用: 家庭・倉庫での片付け / 仕分け、言語で指示できる協働ロボット",
    ], 8.0)
    s2.close()

    concat = out_dir / "concat.txt"
    concat.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    os.system(f"ffmpeg -y -loglevel error -f concat -safe 0 -i {concat} -c copy {video}")
    concat.unlink()
    for p in parts:
        p.unlink()
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\nvideo: {video}\nscore: {total_ok}/{total}")


def single(instruction: str, seed: int, n_objects: int, video: Path | None):
    rng = np.random.default_rng(seed)
    combos = [ALL_COMBOS[i] for i in rng.choice(len(ALL_COMBOS), n_objects, replace=False)]
    sess = Session(random_layout(rng, combos), Perception(), video)
    result = run_scenario(sess, "カスタム指示", instruction, make_parser())
    sess.close()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="record the two-scenario submission video")
    ap.add_argument("-i", "--instruction", default="テーブルの上のものを同じ色のトレイに片付けて")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-n", "--objects", type=int, default=5)
    ap.add_argument("--video", type=Path, default=None, help="optional mp4 path for single runs")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    if args.demo:
        demo(args.out)
    else:
        single(args.instruction, args.seed, args.objects, args.video)


if __name__ == "__main__":
    main()
