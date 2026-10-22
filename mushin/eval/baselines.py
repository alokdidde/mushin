"""M2 baselines on a generated dataset, scored with the same metrics as the model.

    python -m mushin.eval.baselines DATA_DIR [--out results/baselines_m0.json]

- ``rule``: uzumaki's rule selector (the teacher) with label smoothing fitted on ``calib``;
  P(success) is the training success rate of the option's skill.
- ``frequency``: how often each option is the target in ``train``, given the protocol, gripper
  state and the previous step's option and tool results.
- ``linear``: a conditional-logit choice model and a logistic P(success) model over crossed
  one-hot features of the decision-time state (observation fields, last step, option), trained
  on ``train`` with L2. No frames.

The LLM-planner baseline (uzumaki's ``ModelPlanner``) is run separately when a provider is
configured; it is not part of this file's numbers.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from mushin.data.splits import load, read_records
from mushin.eval.metrics import choice_metrics, success_metrics
from mushin.schema import ABSTAIN

Record = dict[str, Any]


def state_features(record: Record) -> list[str]:
    obs = record["observation"]
    feats = [f"protocol={record['protocol']}", f"gripper={obs.get('gripper_state')}"]
    if "holder" in obs:
        feats.append(f"holder={obs.get('holder')}")
    reach = obs.get("within_reach")
    if isinstance(reach, dict):
        feats += [f"reach_object={reach.get('object')}", f"reach_target={reach.get('target')}"]
    last = record["history"][-1] if record["history"] else None
    feats.append(f"last={last['option_id'] if last else 'none'}")
    if last:
        for result in last["tool_results"]:
            feats.append(f"last_tool={result['tool']}:{result['status']}")
    feats.append(f"steps={min(len(record['history']), 3)}")
    return feats


def option_features(record: Record, option: str) -> list[str]:
    skill = option.split("@")[0]
    base = [f"option={skill}"]
    return base + [f"{skill}|{f}" for f in state_features(record)]


def context_key(record: Record) -> str:
    return "|".join(state_features(record)[:5])


class Vocabulary:
    def __init__(self) -> None:
        self.index: dict[str, int] = {}

    def fit(self, rows: Iterable[list[str]]) -> None:
        for row in rows:
            for feature in row:
                self.index.setdefault(feature, len(self.index))

    def vector(self, row: list[str]) -> np.ndarray:
        out = np.zeros(len(self.index))
        for feature in row:
            if feature in self.index:
                out[self.index[feature]] = 1.0
        return out


def train_conditional_logit(records: list[Record], vocab: Vocabulary, l2: float = 1e-2,
                            steps: int = 300, lr: float = 0.5) -> np.ndarray:
    weights = np.zeros(len(vocab.index))
    items = [(np.stack([vocab.vector(option_features(r, o["option_id"])) for o in r["options"]]),
              np.array([r["labels"]["choice_soft"][o["option_id"]] for o in r["options"]]))
             for r in records]
    for _ in range(steps):
        grad = l2 * weights
        for x, soft in items:
            z = x @ weights
            p = np.exp(z - z.max())
            p /= p.sum()
            grad += x.T @ (p - soft) / len(items)
        weights -= lr * grad
    return weights


def train_logistic(rows: list[np.ndarray], labels: list[float], l2: float = 1e-2,
                   steps: int = 300, lr: float = 0.5) -> np.ndarray:
    x, y = np.stack(rows), np.array(labels)
    weights = np.zeros(x.shape[1] + 1)
    xb = np.hstack([x, np.ones((len(x), 1))])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(xb @ weights)))
        weights -= lr * (xb.T @ (p - y) / len(y) + l2 * np.r_[weights[:-1], 0])
    return weights


def score(records: list[Record], choice: Callable[[Record], list[float]],
          success: Callable[[Record, str], float]) -> dict[str, Any]:
    probs, hard, soft, s_probs, s_labels = [], [], [], [], []
    for r in records:
        ids = [o["option_id"] for o in r["options"]]
        probs.append(choice(r))
        hard.append(ids.index(r["labels"]["choice"]))
        soft.append([r["labels"]["choice_soft"][o] for o in ids])
        for o in ids:
            label = r["labels"]["option_outcomes"][o]["success"]
            if label is not None:
                s_probs.append(success(r, o))
                s_labels.append(float(label))
    return {"choice": choice_metrics(probs, hard, soft),
            "success": success_metrics(s_probs, s_labels)}


def run(data: Path) -> dict[str, Any]:
    owner = load(data)
    splits: dict[str, list[Record]] = defaultdict(list)
    for record in read_records(data / "records.jsonl"):
        splits[owner[record["episode_id"]]].append(record)
    train, calib = splits["train"], splits.get("calib", [])

    skill_rate: dict[str, list[float]] = defaultdict(list)
    for r in train:
        for o, out in r["labels"]["option_outcomes"].items():
            if out["success"] is not None:
                skill_rate[o.split("@")[0]].append(float(out["success"]))
    overall = float(np.mean([v for vs in skill_rate.values() for v in vs]))
    rate = {k: float(np.mean(v)) for k, v in skill_rate.items()}

    def prior_success(r: Record, o: str) -> float:
        return rate.get(o.split("@")[0], overall)

    def rule_probs(r: Record, alpha: float) -> list[float]:
        ids = [o["option_id"] for o in r["options"]]
        n = len(ids)
        return [(1 - alpha) * (o == r["teacher"]["option_id"]) + alpha / n for o in ids]

    alphas = [0.001, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5]

    def nll(alpha: float) -> float:
        return float(np.mean([-math.log(rule_probs(r, alpha)[
            [o["option_id"] for o in r["options"]].index(r["labels"]["choice"])])
            for r in calib])) if calib else 0.0

    alpha = min(alphas, key=nll)

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for r in train:
        counts[context_key(r)][r["labels"]["choice"].split("@")[0]] += 1

    def frequency_probs(r: Record) -> list[float]:
        seen = counts.get(context_key(r), Counter())
        raw = [seen[o["option_id"].split("@")[0]] + 0.5 for o in r["options"]]
        return [x / sum(raw) for x in raw]

    vocab = Vocabulary()
    vocab.fit(option_features(r, o["option_id"]) for r in train for o in r["options"])
    weights = train_conditional_logit(train, vocab)

    def linear_probs(r: Record) -> list[float]:
        z = np.array([vocab.vector(option_features(r, o["option_id"])) @ weights
                      for o in r["options"]])
        p = np.exp(z - z.max())
        return list(p / p.sum())

    rows, labels = [], []
    for r in train:
        for o, out in r["labels"]["option_outcomes"].items():
            if out["success"] is not None:
                rows.append(vocab.vector(option_features(r, o)))
                labels.append(float(out["success"]))
    s_weights = train_logistic(rows, labels)

    def linear_success(r: Record, o: str) -> float:
        z = np.r_[vocab.vector(option_features(r, o)), 1.0] @ s_weights
        return float(1 / (1 + math.exp(-z)))

    baselines = {
        "rule": (lambda r: rule_probs(r, alpha), prior_success),
        "frequency": (frequency_probs, prior_success),
        "linear": (linear_probs, linear_success),
    }
    results: dict[str, Any] = {
        "data_records_sha256": json.loads((data / "splits.json").read_text())["records_sha256"],
        "rule_smoothing": alpha,
        "splits": {k: len(v) for k, v in sorted(splits.items())},
        "teacher_matches_target": {
            k: float(np.mean([r["teacher"]["option_id"] == r["labels"]["choice"] for r in v]))
            for k, v in sorted(splits.items())},
        "abstain_target_rate": {
            k: float(np.mean([r["labels"]["choice"] == ABSTAIN for r in v]))
            for k, v in sorted(splits.items())},
        "baselines": {},
    }
    for name, (choice, success) in baselines.items():
        results["baselines"][name] = {
            split: score(records, choice, success)
            for split, records in sorted(splits.items()) if split != "train"
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mushin.eval.baselines", description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    options = parser.parse_args(argv)
    results = run(options.data)
    text = json.dumps(results, indent=1)
    if options.out:
        options.out.parent.mkdir(parents=True, exist_ok=True)
        options.out.write_text(text + "\n")
    for name, splits in results["baselines"].items():
        print(name, {s: (round(m["choice"]["accuracy"], 3), round(m["choice"]["ece"], 3),
                         round(m["success"].get("brier", math.nan), 3))
                     for s, m in splits.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
