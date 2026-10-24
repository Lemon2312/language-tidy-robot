"""Natural-language instruction -> structured tidy-up plan.

Two back-ends:
* ``RuleParser``: offline, deterministic parser for Japanese / English (default).
* ``LLMParser``: optional, used when OPENAI_API_KEY is set; falls back to the rule parser on any error.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import urllib.request

SAME = "same"

COLOR_WORDS = {
    "red": ["赤", "あか", "red"],
    "blue": ["青", "あお", "blue"],
    "green": ["緑", "みどり", "green"],
    "yellow": ["黄色", "黄", "きいろ", "yellow"],
}
SHAPE_WORDS = {
    "cube": ["キューブ", "ブロック", "立方体", "積み木", "cube", "block"],
    "ball": ["ボール", "球", "玉", "ball", "sphere"],
    "cylinder": ["円柱", "シリンダー", "筒", "缶", "cylinder", "can"],
}
CONTAINER = r"(?:トレイ|トレー|箱|はこ|皿|ボックス|かご|tray|box|bin|basket)"
SAME_COLOR = r"(?:同じ色|おなじ色|同色|色ごと|色別|色で分け|same[- ]colou?r|matching|by colou?r)"
CLAUSE_SPLIT = r"[。．\.;；\n]|、(?:そして|それから|次に|その後)?|そして|それから|次に|その後|,\s*(?:and\s+)?then|\bthen\b"
CONJ_SPLIT = r"と|や|および|及び|,|\band\b"
_COLOR_RE = "|".join(sorted((re.escape(w) for ws in COLOR_WORDS.values() for w in ws), key=len, reverse=True))
DEST_RE = rf"({_COLOR_RE})(?:色の|い|の)?\s*{CONTAINER}"


@dataclasses.dataclass
class Target:
    color: str | None = None
    shape: str | None = None

    def matches(self, color: str, shape: str) -> bool:
        return (self.color is None or self.color == color) and (self.shape is None or self.shape == shape)

    def describe_ja(self) -> str:
        c = {"red": "赤い", "blue": "青い", "green": "緑の", "yellow": "黄色い"}.get(self.color or "", "")
        s = {"cube": "キューブ", "ball": "ボール", "cylinder": "円柱"}.get(self.shape or "", "もの")
        if not c and s == "もの":
            return "すべての物体"
        return f"{c}{s}"


@dataclasses.dataclass
class Step:
    target: Target
    dest: str  # tray colour, or SAME for "the tray with the same colour as the object"

    def describe_ja(self) -> str:
        d = "同じ色のトレイ" if self.dest == SAME else {"red": "赤", "blue": "青", "green": "緑", "yellow": "黄"}[self.dest] + "のトレイ"
        return f"{self.target.describe_ja()} → {d}"


def _find(words: dict[str, list[str]], text: str) -> str | None:
    hits = []
    for key, ws in words.items():
        for w in ws:
            m = re.search(re.escape(w), text, flags=re.IGNORECASE)
            if m:
                hits.append((m.start(), key))
    return min(hits)[1] if hits else None


class RuleParser:
    name = "rule-based parser"

    def parse(self, instruction: str) -> list[Step]:
        steps: list[Step] = []
        for clause in re.split(CLAUSE_SPLIT, instruction, flags=re.IGNORECASE):
            clause = (clause or "").strip()
            if not clause:
                continue
            dest = None
            matches = list(re.finditer(DEST_RE, clause, flags=re.IGNORECASE))
            m = matches[-1] if matches else None
            if re.search(SAME_COLOR, clause, flags=re.IGNORECASE):
                dest = SAME
                clause = re.sub(SAME_COLOR, " ", clause, flags=re.IGNORECASE)
            elif m:
                dest = _find(COLOR_WORDS, m.group(1))
            obj_text = clause[: m.start()] + " " + clause[m.end():] if m and dest != SAME else clause
            if dest is None:
                if re.search(r"片付け|片づけ|仕分け|整理|tidy|sort|clean", clause, flags=re.IGNORECASE):
                    dest = SAME
                else:
                    continue
            targets = [Target(_find(COLOR_WORDS, p), _find(SHAPE_WORDS, p))
                       for p in re.split(CONJ_SPLIT, obj_text, flags=re.IGNORECASE)]
            targets = [t for t in targets if t.color or t.shape] or [Target()]
            steps.extend(Step(t, dest) for t in targets)
        return _dedupe(steps)


def _dedupe(steps: list[Step]) -> list[Step]:
    out: list[Step] = []
    for s in steps:
        if s not in out:
            out.append(s)
    return out


LLM_SYSTEM = """You convert a tidy-up instruction for a robot arm into JSON.
Objects have color in {red, blue, green, yellow} and shape in {cube, ball, cylinder}.
Trays exist in colors red, blue, green.
Return {"steps": [{"color": <color|null>, "shape": <shape|null>, "dest": <tray color|"same">}, ...]}
in the order the user wants them executed. null means "any". "same" means the tray matching the object's color."""


class LLMParser:
    name = "LLM parser"

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.key = os.environ["OPENAI_API_KEY"]
        self.fallback = RuleParser()

    def parse(self, instruction: str) -> list[Step]:
        body = json.dumps({
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": instruction}],
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                content = json.loads(r.read())["choices"][0]["message"]["content"]
            steps = [Step(Target(s.get("color"), s.get("shape")), s.get("dest") or SAME)
                     for s in json.loads(content)["steps"]]
            return steps or self.fallback.parse(instruction)
        except Exception as e:  # network / schema errors: degrade gracefully
            print(f"[language] LLM parse failed ({e}); using rule-based parser")
            return self.fallback.parse(instruction)


def make_parser():
    return LLMParser() if os.environ.get("OPENAI_API_KEY") else RuleParser()
