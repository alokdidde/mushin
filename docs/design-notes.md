# Design notes

The original research framing for Mushin, written before the first model was trained. The [README](../README.md) describes the project as it is now.

## Motivation

Many decisions inside an agent or robot controller are narrow: Is this skill applicable? Which recovery action should be tried? Is the current evidence sufficient to act? Generating a free-form response for every such decision can add latency and complicate integration.

Mushin investigates whether a compact model trained for bounded judgments can make these decisions efficiently while producing probabilities that can be evaluated for calibration. A valid output type alone does not establish that a judgment is correct or that its confidence is calibrated.

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
