# Mushin

**A research project exploring compact System-1 models for typed probabilistic decisions.**

Mushin aims to turn context and a focused question into a bounded decision that software can use directly: a probability, a choice among explicit alternatives, or a score on a defined scale. The research goal is to study the trade-offs between decision quality, calibration, and inference cost, with potential applications to agent routing and robotic skill selection.

## Current status

**[mushin-0.8b-v0.1](https://huggingface.co/alokdidde/mushin-0.8b-v0.1) is trained and benchmarked** (research preview, simulation only). It is Qwen3.5-0.8B with LoRA adapters and typed decision heads, trained on 201k simulated robot decisions with counterfactual labels. Given the task, the observation, recent history and scene + wrist camera frames, it picks one of the offered skills with its parameters, or abstains, and estimates P(success) for every option, in one forward pass.

- **Offline:** it beats every baseline on choice accuracy (test 0.747 vs 0.715 for the best baseline; 0.707 vs 0.632 on a robot body it never saw) and on P(success) ranking (AUROC 0.916 vs 0.831), with well-calibrated P(success) (ECE 0.005).
- **Weakness:** its choice probabilities are overconfident (ECE 0.16 after temperature scaling), so the frequency baseline still has a better choice Brier score and risk-coverage curve.
- **Closed loop:** as uzumaki's planner over 480 paired sim episodes it matches the rule planner's task success within noise (76.9% vs 78.1%) with 17% fewer decisions, and abstains on unreachable targets.
- **Latency:** 79 ms median forward pass on an A40 (bf16, batch 1, 1.98 GB). The Jetson Orin Nano target has not been measured yet.

Hugging Face: model [alokdidde/mushin-0.8b-v0.1](https://huggingface.co/alokdidde/mushin-0.8b-v0.1), datasets [alokdidde/mushin-sim-decisions-v1](https://huggingface.co/datasets/alokdidde/mushin-sim-decisions-v1) (training data) and [alokdidde/mushin-sim-decisions-v2](https://huggingface.co/datasets/alokdidde/mushin-sim-decisions-v2) (v1 + a G1 top-up). These repos are private for now.

| Milestone | Status |
|---|---|
| M0 decision records + sim data generator | done |
| M1 model and training code | done |
| M2 baselines | done |
| M3 RunPod launcher, latency tool, uzumaki adapter | done |
| M4 dataset (201k decisions, v1; 206k, v2) | done |
| M5 SFT: head warm-up, LoRA, calibration | done (mushin-0.8b-v0.1) |
| Offline + closed-loop benchmarks | done |
| M6 episode RL with a calibration reward | planned |
| M7 ONNX/TensorRT export and Jetson Orin Nano latency | planned |

## Results

All numbers are on held-out splits of [mushin-sim-decisions-v1](https://huggingface.co/datasets/alokdidde/mushin-sim-decisions-v1), after temperature calibration on the separate `calib` split. Choice accuracy is measured against the counterfactual best option (see [Data generation](#data-generation-m0)), not against the teacher's action. `test_body` is the Go1 arm, a whole robot body held out of training. AURC is the area under the risk-coverage curve when the least confident decisions are deferred first.

| split | model | choice acc ↑ | choice Brier ↓ | AURC ↓ | success AUROC ↑ | success ECE ↓ |
|---|---|---:|---:|---:|---:|---:|
| val (n=8,771) | **Mushin 0.8B v0.1** | **0.756** | 0.448 | 0.153 | **0.916** | 0.005 |
| | frequency | 0.725 | **0.368** | **0.124** | 0.609 | 0.004 |
| | linear | 0.717 | 0.507 | 0.196 | 0.831 | 0.075 |
| | rule selector (data teacher) | 0.632 | 0.553 | 0.312 | 0.609 | 0.004 |
| test (n=8,997) | **Mushin 0.8B v0.1** | **0.747** | 0.456 | 0.166 | **0.916** | 0.005 |
| | frequency | 0.715 | **0.376** | **0.131** | 0.613 | 0.006 |
| | linear | 0.709 | 0.515 | 0.206 | 0.831 | 0.081 |
| | rule selector | 0.627 | 0.559 | 0.335 | 0.613 | 0.006 |
| test_body, Go1 (n=23,453) | **Mushin 0.8B v0.1** | **0.707** | 0.520 | **0.144** | **0.877** | 0.010 |
| | frequency | 0.632 | **0.464** | 0.208 | 0.538 | 0.096 |
| | linear | 0.630 | 0.620 | 0.215 | 0.716 | 0.063 |
| | rule selector | 0.442 | 0.745 | 0.554 | 0.538 | 0.096 |

The head warm-up checkpoint alone (backbone frozen) reached 0.708 choice accuracy and 0.833 success AUROC on val; LoRA adds the rest.

**Per body (test choice accuracy):** ALOHA 0.802, Go1 walking 0.802, TidyBot 0.791, Panda 0.706, Spot 0.704, SO-101 0.701, G1 0.665. G1 was under-represented in v1, which the v2 dataset fixes.

**Calibration:** fitted temperatures are 0.780 (choice) and 1.065 (success). Choice ECE goes from 0.184 to 0.158 on `calib`; it stays at 0.16 on val/test and 0.20 on Go1, so a single temperature is not enough. P(success) ECE is 0.005-0.010.

**Closed loop** (`python -m mushin.eval.closed_loop`): whole uzumaki episodes on fresh training-only seeds with random faults, 60 per body, once with the rule planner and once with Mushin on the same cases; uzumaki's evaluator decides success.

| body | rule success | Mushin success | decisions / episode, Mushin (rule) |
|---|---:|---:|---:|
| **all (480)** | **0.781** | **0.769** | **2.44 (2.95)** |
| SO-101 | 0.733 | 0.733 | 1.57 (2.42) |
| Panda | 0.750 | 0.750 | 1.57 (2.53) |
| Spot | 0.833 | 0.833 | 1.55 (2.40) |
| G1 | 0.800 | 0.767 | 1.50 (2.40) |
| Go1 (unseen body) | 0.850 | 0.850 | 1.55 (2.48) |
| ALOHA | 0.817 | 0.817 | 3.30 (3.30) |
| TidyBot | 0.683 | 0.633 | 4.17 (3.95) |
| Go1 walking | 0.783 | 0.767 | 4.30 (4.10) |

Paired: both succeed 368, neither 104, rule only 7, Mushin only 1. Most shared failures come from injected faults (budget exhausted 55, unreachable target 22, cancellations, evaluation failures). Mushin abstains on the 22 unreachable targets where the rule planner tries and fails (both count as failures), and has 3 more collisions (TidyBot) and 2 missing-observation stops. Decision time inside the sim (median 248 ms) includes rendering the camera frames.

**Latency** (`python -m mushin.export.latency`, 50 real records, batch 1, ~532 tokens of which 128 are image tokens): A40 bf16 forward median 79 ms, p90 108 ms, preprocessing 11 ms, peak memory 1.98 GB. This is without `causal-conv1d` (its build failed against the pod's CUDA), so Qwen3.5's linear-attention layers ran on the slower PyTorch path. The Jetson Orin Nano has not been measured.

All result files (per-split and per-body metrics, calibration, latency, closed-loop episodes, training curve) are in the model repo's [`results/`](https://huggingface.co/alokdidde/mushin-0.8b-v0.1/tree/main/results).

### Limitations

- Simulation only: software-rendered MuJoCo frames and uzumaki's simulated skills. Never run on a real robot.
- Overconfident choice probabilities (see above). Don't threshold abstention on them without your own calibration check.
- Closed-loop parity with the rule planner, not improvement, so far.
- Three task protocols, a small skill library, and one held-out body; unseen tasks are untested.
- Labels are counterfactual outcomes under uzumaki's simulator with the rule selector finishing each episode, so they encode that simulator and that teacher.

## Use the model

```bash
uv venv -p 3.12 && uv pip install -e ".[train]"      # torch, transformers>=5.20, peft, safetensors
uv pip install -e ".[gpu]" causal-conv1d             # optional on CUDA: fast linear-attention kernels
```

```python
from pathlib import Path
import torch
from mushin.train import checkpoint
from mushin.train.data import load_dataset, make_example

model = checkpoint.load_pretrained("alokdidde/mushin-0.8b-v0.1", device="cuda")
data = Path("data/m4")                                   # a mushin.decision/1.0 dataset directory
records, cards = load_dataset(data, splits=["test"], limit=1)
example = make_example(records[0], cards, model.backbone, data)
with torch.inference_mode():
    pred = model(model.backbone.collate([example.encoded]), [example.layout])[0]
options = example.layout.option_ids
print(dict(zip(options, torch.softmax(pred.choice.float(), -1).tolist())))   # choice, incl. abstain
print(dict(zip(options, torch.sigmoid(pred.success.float()).tolist())))      # P(success) per skill
```

`load_pretrained` takes a Hub repo id or a local folder with `mushin_config.json` and `model.safetensors`, and builds the Qwen3.5-0.8B base from its own Hub repo. In uzumaki, wrap the model in `mushin.uzumaki_adapter.MushinPlanner` (a `DecisionSource`). `python -m mushin.export.release CHECKPOINT OUT_DIR` turns a training checkpoint into a release folder.

## Datasets

| Dataset | Decisions | Episodes | Notes |
|---|---:|---:|---|
| [mushin-sim-decisions-v1](https://huggingface.co/datasets/alokdidde/mushin-sim-decisions-v1) | 201,078 | 75,667 | Training and evaluation data of mushin-0.8b-v0.1 |
| [mushin-sim-decisions-v2](https://huggingface.co/datasets/alokdidde/mushin-sim-decisions-v2) | 206,262 | 77,782 | v1 + 2,144 G1 episodes; v1 episodes keep their splits; for the next model |
| [mushin-sim-decisions-g1-topup](https://huggingface.co/datasets/alokdidde/mushin-sim-decisions-g1-topup) | 5,184 | 2,144 | The G1 top-up on its own (provenance) |

- 8 bodies (SO-101, Panda, Spot, G1, Go1 arm on `pick-place-v1`; ALOHA on `handover-v1`; TidyBot and Go1 walking on `fetch-v1`), uzumaki commit `2bf3a334`, seeds `mushin-train-*` only.
- Splits (v1, decisions): train 150,970, val 8,771, calib 8,887, test 8,997, test_body (Go1) 23,453.
- Best-choice targets (v1): place-box 45k, pick-box 44k, abstain 41k, pick-and-place 37k, go-to 25k, hand-over 8k. The rule teacher agrees with the counterfactual best option only 60.8% of the time, which is why labels come from counterfactuals.
- About 2.9 GB of 224 px JPEG frames, stored in `shards/*/images.tar` (unpack with `python -m mushin.data.finalize`). Generation cost about $12 on a 32-vCPU RunPod CPU machine.
- No public robot datasets were used for v0.1. Licences for the data and model are not chosen yet; the Qwen3.5-0.8B base is Apache-2.0, and the robot models come from upstream asset packs with their own licences.

## Motivation

Many decisions inside an agent or robot controller are narrow: Is this skill applicable? Which recovery action should be tried? Is the current evidence sufficient to act? Generating a free-form response for every such decision can add latency and complicate integration.

Mushin investigates whether a compact model trained for bounded judgments can make these decisions efficiently while producing probabilities that can be evaluated for calibration. A valid output type alone does not establish that a judgment is correct or that its confidence is calibrated.

The project is inspired by [TypeSafe AI’s Jev and its System One model framing](https://typesafe.ai/blog/introducing-system-one-models-and-jev). Mushin is an independent project, with no affiliation to TypeSafe AI and no claim to reproduce Jev’s proprietary architecture, training method, or reported performance.

## How it is built

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

### Model and training (M1)

`mushin.model` follows the published Valen/agent-jev recipe in Mushin's own code. The backbone is Qwen3.5-0.8B (`Qwen3_5Model`: vision tower, connector and language model) with its LM head removed. Each decision is one shared-state pass over the task card, observation, recent steps, scene and wrist frames, and every option with its skill card. Typed heads read the hidden states at marked tokens:

- **choice:** dot product between the final `Decide` token and each option's last token, abstain included;
- **yes/no:** P(success) for each skill option;
- **value bins / enums:** a distribution over bins for each number parameter and over the allowed values of each enum.

Training (`mushin.train.sft`) has two stages. First a warm-up on the heads with the backbone frozen, then LoRA on the language model plus the vision connector. The objective is soft cross-entropy plus 0.1 Brier on the choice, BCE on success, and CE on parameters. Options are shuffled every epoch. `mushin.train.calibrate` fits the choice and success temperatures on the calibration split, and `mushin.eval.run` reports accuracy, NLL, Brier, ECE and risk-coverage per split and per body.

mushin-0.8b-v0.1 used LoRA rank 16, alpha 32, dropout 0.05 on the attention, Gated DeltaNet and MLP projections, with the vision merger trained in full (24.4M trainable parameters); AdamW at 1e-4 (adapters) and 3e-4 (heads), effective batch 16, 9,436 warm-up steps then 18,872 LoRA steps (2 epochs), bf16 on RunPod A40/L40S GPUs, about $13 in total.

```bash
uv pip install -e ".[train,dev]"
python -m mushin.train.sft --config configs/sft_tiny.yaml            # CPU smoke run
python -m mushin.train.sft --config configs/sft_qwen35_08b_warmup.yaml
python -m mushin.train.sft --config configs/sft_qwen35_08b_lora.yaml
python -m mushin.train.calibrate --checkpoint runs/qwen35_08b_lora/checkpoint.pt --data data/m4
```

### Baselines (M2)

`python -m mushin.eval.baselines DATA_DIR --out results/baselines_m0.json` scores three baselines with the model's metrics: uzumaki's rule selector with fitted smoothing, a frequency table, and a linear conditional-logit model over decision-time features. Full-dataset numbers are in [Results](#results). Frozen numbers for the small M0 sample set (2,066 decisions, 800 episodes, Go1 held out) are in [`results/`](results/). Choice accuracy against the counterfactual target on that sample set:

| Baseline | val | test | test_body (Go1) |
| --- | --- | --- | --- |
| rule selector | 0.687 | 0.670 | 0.469 |
| frequency | 0.707 | 0.795 | 0.680 |
| linear | 0.717 | 0.773 | 0.689 |

The linear model's P(success) reaches AUROC 0.81 on val, and the M1 gate run (`configs/sft_tiny.yaml`) fits 200 records to within 0.03 nats of the soft target ([`results/m1_overfit_tiny.json`](results/m1_overfit_tiny.json)).

### RunPod, latency and uzumaki (M3)

- `python -m mushin.runpod.launch configs/stages/<stage>.yaml --dry-run` ranks the live RunPod offers that fit the stage by estimated total cost (price per hour times estimated hours, using measured throughput once a benchmark exists), then prints the choice and its cost cap. A real start needs `--approve-cap` at or above that cap. The launcher stops the pod when the job ends, when the cost reaches the cap, or at `max_hours`, and the pod also stops itself (`scripts/pod_job.sh`). `configs/stages/offers_example.json` holds illustrative prices for offline tests, not live quotes.
- `python -m mushin.export.latency` times preprocessing and the forward pass at batch size 1 on real records. ONNX/TensorRT export for the Jetson Orin Nano is milestone M7.
- `python -m mushin.eval.closed_loop --checkpoint CKPT --out DIR --episodes 60` runs paired uzumaki episodes with the rule selector and with `MushinPlanner`.
- `mushin.uzumaki_adapter.MushinPlanner` is an uzumaki `DecisionSource`. It renders frames through the sim frame hook, abstains when abstain wins or confidence is low, and returns decisions in the rule selector's shape.

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

## Artifacts

- Training and inference code with versioned decision schemas (this repo).
- Dataset preparation and evaluation scripts with provenance and split definitions (this repo), and the datasets on Hugging Face (see [Datasets](#datasets)).
- The model and its model card, covering intended uses, limitations and measured results: [alokdidde/mushin-0.8b-v0.1](https://huggingface.co/alokdidde/mushin-0.8b-v0.1).

There is no hosted endpoint.

## Related work

- Mohamad Yazan Sadoun, Sarah Sharif, and Yaser Mike Banad. **[From Probabilities to Decisions: Search and Multi-Teacher Distillation with Jev](https://arxiv.org/abs/2610.09188)**. arXiv preprint, 2026. Studies compact evaluators distilled from Jev judgments and multi-teacher distillation, providing related work for Mushin’s interest in efficient probabilistic decisions. This reference does not imply that Mushin has implemented or reproduced the paper’s methods or results.
