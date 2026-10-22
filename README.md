# Mushin

**A research project exploring compact System-1 models for typed probabilistic decisions.**

Mushin aims to turn context and a focused question into a bounded decision that software can use directly: a probability, a choice among explicit alternatives, or a score on a defined scale. The research goal is to study the trade-offs between decision quality, calibration, and inference cost, with potential applications to agent routing and robotic skill selection.

## Current status

Milestone M0 (data) is implemented: a versioned decision record format and a generator that produces labelled decisions with camera frames from [uzumaki](https://github.com/alokdidde/uzumaki)'s MuJoCo scenes. Model code, training, released weights and benchmark results are not yet available; the interfaces further below illustrate the intended design and are not a working API.

### Data generation (M0)

Mushin's first target is uzumaki's planner slot (`DecisionSource`): at each decision it sees the task goal, the latest observation, recent history and scene + wrist camera frames, and chooses one eligible skill with typed parameters, or abstains. `mushin.data` produces training records for exactly that:

- **Bodies and task packs:** all eight uzumaki bodies (SO-101, Panda, Spot, G1, Go1 on `pick-place-v1`; ALOHA on `handover-v1`; TidyBot and the walking Go1 on `fetch-v1`), on fresh training-only seeds (`mushin-train-*`) with random injected faults (misgrasps, stale observations, dropped handovers, short drives, crashes, goal withdrawal, unreachable targets).
- **Exploration:** the rule selector acts as teacher; with probability epsilon a random offered option (or abstain) is taken instead, with oracle or random parameters.
- **Counterfactual labels:** every decision is replayed and each offered skill is tried in its place, with the rule selector finishing the episode. This gives P(success) labels for every option, a hard and soft choice target (fewer decisions preferred, abstain when nothing succeeds), and parameter targets (value bins for numbers, per-value success for enums). The simulator is deterministic, so each label carries the evidence-bundle hash that reproduces it.
- **Frames:** 224 px scene and wrist JPEGs per decision, with render-only domain randomization (lighting, colours, camera pose) that leaves physics and labels unchanged.
- **Splits:** whole held-out bodies (default: Go1) and task packs, then a seed-hash split into train / val / calib / test; `splits.json` pins the records hash.

```bash
uv venv -p 3.12 && uv pip install -e ".[sim,dev]"
scripts/fetch_uzumaki.sh                      # pinned checkout in third_party/ + MuJoCo assets
python -m mushin.data.generate --out data/m0 --episodes 800 --workers 8
python -m mushin.data.splits data/m0 --config configs/data_m0.yaml
python -m mushin.data.verify data/m0 --reproduce 50
```

The record format is `mushin.decision/1.0` ([schema](mushin/schema/decision-1.0.schema.json)); skill and task cards are stored once per dataset in `cards.json`. Rendering needs EGL or OSMesa (`MUJOCO_GL=egl`).

## Motivation

Many decisions inside an agent or robot controller are narrow: Is this skill applicable? Which recovery action should be tried? Is the current evidence sufficient to act? Generating a free-form response for every such decision can add latency and complicate integration.

Mushin will investigate whether a compact model trained for bounded judgments can make these decisions efficiently while producing probabilities that can be evaluated for calibration. A valid output type alone does not establish that a judgment is correct or that its confidence is calibrated.

The project is inspired by [TypeSafe AI’s Jev and its System One model framing](https://typesafe.ai/blog/introducing-system-one-models-and-jev). Mushin is an independent project, with no affiliation to TypeSafe AI and no claim to reproduce Jev’s proprietary architecture, training method, or reported performance.

## Intended decision interface

The proposed contract is **context + typed question → structured prediction**. Initial experiments will focus on bounded classification before considering broader scoring tasks.

| Decision type | Intended output | Example application |
| --- | --- | --- |
| Binary judgment | Probability of a defined event | Is a candidate skill applicable? |
| Choice | Probability distribution over explicit alternatives | Which recovery action should be selected? |
| Bounded score | Estimate on a specified scale | How relevant is an observation to a task? |

Illustrative input:

```json
{
  "context": {
    "task": "pick_and_place",
    "last_outcome": "failed_grasp",
    "object_visible": true
  },
  "question": {
    "type": "choice",
    "prompt": "Which recovery action should be selected?",
    "options": ["reobserve", "retry_grasp", "abstain"]
  }
}
```

Illustrative output (**synthetic values, not model predictions**):

```json
{
  "probabilities": {
    "reobserve": 0.65,
    "retry_grasp": 0.20,
    "abstain": 0.15
  }
}
```

Application code would apply separately evaluated thresholds, permissions, and execution budgets. Predicted success would not replace independent outcome verification.

## Proposed approach

1. Define a narrow decision task, input schema, labels, data provenance, and frozen evaluation splits.
2. Establish simple rule-based, conventional classifier, and prompted language-model baselines under the same task definition.
3. Train a compact decision model using suitable public or explicitly authorized data; document the architecture, training objective, and data sources when selected.
4. Evaluate probability calibration and abstention separately from decision accuracy.
5. Explore integration with [Uzumaki](https://github.com/alokdidde/uzumaki), initially through offline fixtures and subsequently through simulator-backed experiments if the standalone results support it.

Architecture, datasets, model size, and training recipe remain research choices. No particular speedup, calibration level, or downstream robotics benefit is assumed.

## Evaluation plan

- **Decision quality:** accuracy and task-appropriate classification metrics.
- **Probability quality:** negative log-likelihood, Brier score, and reliability diagrams; report any calibration fitting on a separate validation split.
- **Selective prediction:** accuracy or risk as a function of coverage when uncertain cases are deferred.
- **Efficiency:** median and tail latency, throughput, and memory use with hardware, batch size, and input length disclosed.
- **Generalization:** held-out task instances and condition shifts, with related examples kept out of both training and evaluation where applicable.
- **Downstream utility:** compare task completion and recovery behavior under matched retry and resource budgets before claiming a benefit to Uzumaki.

Results will distinguish standalone model judgments from closed-loop system performance. Reproducible configurations, baseline definitions, and limitations will accompany any reported measurements.

## Planned artifacts

- Training and inference code with versioned decision schemas.
- Dataset preparation and evaluation scripts with provenance and split definitions.
- A model card covering intended uses, limitations, and measured results.
- A Hugging Face release if distributable model weights become available.

There is currently no installation command, hosted endpoint, downloadable checkpoint, or Hugging Face release.

## Related work

- Mohamad Yazan Sadoun, Sarah Sharif, and Yaser Mike Banad. **[From Probabilities to Decisions: Search and Multi-Teacher Distillation with Jev](https://arxiv.org/abs/2610.09188)**. arXiv preprint, 2026. Studies compact evaluators distilled from Jev judgments and multi-teacher distillation, providing related work for Mushin’s interest in efficient probabilistic decisions. This reference does not imply that Mushin has implemented or reproduced the paper’s methods or results.
