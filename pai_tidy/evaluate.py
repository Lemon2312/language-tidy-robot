"""Quantitative evaluation on random scenes (no video).

    python -m pai_tidy.evaluate --perception 30 --tasks 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .language import RuleParser  # noqa: E402
from .perception import Camera, Perception  # noqa: E402
from .robot import OBSERVE_Q  # noqa: E402
from .run import ALL_COMBOS, Session, evaluate  # noqa: E402
from .scene import DEFAULT_TRAYS, random_layout, write_scene  # noqa: E402


def random_objects(rng, n):
    combos = [ALL_COMBOS[i] for i in rng.choice(len(ALL_COMBOS), n, replace=False)]
    return random_layout(rng, combos)


def perception_eval(perception: Perception, n_scenes: int, n_objects: int = 6) -> dict:
    owl_ok = hyb_ok = total = missed = 0
    obj_total = obj_owl_ok = 0
    pos_err, yaw_err = [], []
    for seed in range(n_scenes):
        rng = np.random.default_rng(1000 + seed)
        objs = random_objects(rng, n_objects)
        model = mujoco.MjModel.from_xml_path(str(write_scene(objs, DEFAULT_TRAYS)))
        data = mujoco.MjData(model)
        data.qpos[:7] = OBSERVE_Q
        mujoco.mj_forward(model, data)
        cam = Camera(model, "overhead", 960, 720)
        rgb, _, xyz = cam.capture(data)
        cam.close()
        dets = perception.detect(rgb, xyz)
        gt = [(o.label, np.array(o.pos), o.yaw, o.shape) for o in objs] + \
             [(t.label, np.array(t.pos), 0.0, "tray") for t in DEFAULT_TRAYS]
        for label, pos, yaw, shape in gt:
            total += 1
            near = [d for d in dets if np.linalg.norm(d.position[:2] - pos) < (0.04 if shape == "tray" else 0.02)]
            if not near:
                missed += 1
                continue
            d = min(near, key=lambda d: np.linalg.norm(d.position[:2] - pos))
            owl_ok += d.owl_label == label
            if shape != "tray":
                obj_total += 1
                obj_owl_ok += d.owl_label == label
            hyb_ok += d.label == label
            if shape != "tray":
                pos_err.append(np.linalg.norm(d.position[:2] - pos) * 1000)
            if shape == "cube":
                e = (np.rad2deg(d.yaw - yaw) + 45) % 90 - 45
                yaw_err.append(abs(e))
    return {
        "scenes": n_scenes, "items": total, "detected": total - missed,
        "label_acc_owl_only": round(owl_ok / total, 3), "label_acc_hybrid": round(hyb_ok / total, 3),
        "label_acc_owl_only_objects": round(obj_owl_ok / obj_total, 3),
        "pos_err_mm_mean": round(float(np.mean(pos_err)), 2), "pos_err_mm_max": round(float(np.max(pos_err)), 2),
        "cube_yaw_err_deg_mean": round(float(np.mean(yaw_err)), 2),
    }


def task_eval(perception: Perception, n_trials: int, n_objects: int = 5) -> dict:
    instructions = ["テーブルの上のものを同じ色のトレイに片付けて",
                    "ボールは全部青いトレイに入れて、それから残りを色ごとに片付けて",
                    "Put the cubes into the green tray, then sort the rest by color."]
    parser = RuleParser()
    ok = total = full = 0
    trials = []
    for i in range(n_trials):
        rng = np.random.default_rng(2000 + i)
        sess = Session(random_objects(rng, n_objects), perception, None)
        sess.agent.log = lambda *_: None
        instr = instructions[i % len(instructions)]
        steps = parser.parse(instr)
        st = sess.agent.run(steps)
        res = evaluate(sess.model, sess.data, sess.objects, steps)
        sess.close()
        ok += res["success"]
        total += res["total"]
        full += res["success"] == res["total"]
        trials.append({"seed": 2000 + i, "instruction": instr, "success": res["success"], "total": res["total"],
                       "picks": st.picks, "grasp_failures": st.grasp_failures})
        print(trials[-1])
    return {"trials": n_trials, "object_success_rate": round(ok / total, 3),
            "episode_success_rate": round(full / n_trials, 3), "details": trials}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", type=int, default=30)
    ap.add_argument("--tasks", type=int, default=9)
    ap.add_argument("--out", type=Path, default=Path("outputs/eval.json"))
    args = ap.parse_args()
    perception = Perception()
    result = {"perception": perception_eval(perception, args.perception)}
    print(json.dumps(result, indent=1))
    if args.tasks:
        result["task"] = task_eval(perception, args.tasks)
        print(json.dumps({k: v for k, v in result["task"].items() if k != "details"}, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
