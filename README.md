<div align="center">

<img src="https://raw.githubusercontent.com/NEURAL-Y/cerium-delta/main/public/Cerium-delta.png" alt="Cerium Delta" width="360"/>

# Cerium Delta

**Beyond visualization. Toward measurable neural architecture behavior.**

[![License](https://img.shields.io/badge/License-Apache_2.0-cyan.svg)](https://github.com/NEURAL-Y/cerium-delta/blob/main/LICENSE)
[![release](https://img.shields.io/badge/release-v1.0.0-orange)](https://github.com/NEURAL-Y/cerium-delta/releases/tag/v1.0.0)
[![pre-release](https://img.shields.io/badge/pre--release-v1.1.0-cyan)](https://github.com/NEURAL-Y/cerium-delta/releases/tag/v1.1.0)
[![status](https://raw.githubusercontent.com/NEURAL-Y/cerium-delta/main/public/badge-shimmer.svg)](https://cerium-delta.pages.dev)
![paper](https://raw.githubusercontent.com/NEURAL-Y/cerium-delta/main/public/badge-paper.svg)
[![architecture](https://img.shields.io/badge/architecture-Cerium%20Delta-1f6feb)](https://github.com/NEURAL-Y/cerium-delta/blob/main/docs/ARCHITECTURE.md)
[![Development Charter](https://raw.githubusercontent.com/NEURAL-Y/cerium-delta/main/public/cerium_delta_charter_badge.svg)](https://github.com/NEURAL-Y/cerium-delta/blob/main/docs/Cerium_Delta_Development_Charter.md)

</div>

<br>

<div align="center">
<img src="https://raw.githubusercontent.com/NEURAL-Y/cerium-delta/main/public/sample.png.png" alt="Cerium Delta sample" width="700"/>
</div>

<br>

> Click the release / pre-release badges above to see what shipped and what's currently in development — each links to demo assets and source for that milestone.

---

## What Cerium Delta Is

Cerium Delta is **not** a neural-network drawing tool. Most architecture visualizers answer *"what does the model look like?"* — a static diagram, useful for documentation but silent on behavior.

Cerium Delta asks a different question:

> **How is the model behaving — right now, as it learns?**

It combines model-state extraction, numerical analysis, architecture-aware metrics, and real-time visualization into a single observability layer for neural networks, built on the premise that a model in training is a dynamic system, not a checkpoint.

## The Problem

Standard evaluation stops at the outcome:

```text
Did the loss decrease?
Did accuracy improve?
Did the model generalize?
```

It rarely answers the *process* questions a developer actually needs when something breaks:

```text
Which layer changed?
Which layer stopped contributing?
Where did the signal weaken?
Which neurons became inactive?
Did the architecture learn a useful representation?
Did the model become more or less stable during training?
```

Cerium Delta exists to provide measurements that make these questions answerable.

## A Simple Mental Model

Think of a neural network as a machine with many moving components. Traditional evaluation looks at the machine's final output. Cerium Delta inspects the machine **while it runs** — neuron activity, layer behavior, parameter change, gradient behavior, information propagation — and turns those observations into measurements comparable across layers, training stages, and models.

For researchers and engineers, the same idea is an **observability layer over neural architectures**.

## Neural Activity

Cerium Delta analyzes internal behavior including:

- Neuron activity and inactivity
- Dead / saturated neurons
- Activation distributions
- Gradient propagation
- Weight evolution
- Dropout behavior
- Layer utilization

These are the lower-level observations from which higher-level analysis is built.

## Architecture-Level Analysis

Individual values are rarely enough on their own. Cerium Delta analyzes how components behave **relative to the rest of the architecture** — moving from *"this tensor changed"* to *"this component changed significantly relative to the rest of the architecture."* That distinction is central to the project.

## From Measurements to Meaning

Cerium Delta's current metrics translate internal model state into comparable measures of neural behavior:

| Metric | Purpose |
|---|---|
| **Layer Contribution Score (LCS)** | Measures a layer's relative contribution to the network — not whether it holds large weights, but whether it exhibits meaningful contribution relative to other layers. |
| **Sensitivity** | Measures how strongly a quantity responds to changes in its local or parameterized representation — a perspective distinct from raw magnitude. |
| **Evolution** | Measures how model state changes relative to a reference state across training, turning *"what is the model now?"* into *"how did the model get here?"* |

### Layer Contribution Score (LCS)
Investigates whether a layer contributes meaningfully to the network, using a relative rather than absolute formulation — large weights alone don't imply importance.

### Sensitivity
Measures the responsiveness of a quantity to local or parameter-level perturbation, independent of its raw magnitude.

### Evolution
Tracks state change relative to a reference point across training, making the learning trajectory itself observable rather than only the endpoint.

## Relative Measurement Matters

A large layer naturally has more parameters than a small one — a large raw value doesn't automatically mean a component is more important. Cerium Delta leans on **relative analysis**, including percentile-based comparisons and statistical filtering, so components are judged against an appropriate reference population rather than in isolation.

**Ranking provides context; raw measurements preserve precision.**

## Neural Vitality

A central research direction: is a neural component actively participating in learning, and how does that change over time? LCS, Sensitivity, and Evolution each offer a different lens on the same underlying question — no single metric is treated as a complete verdict; together they form a measurement system rather than a single score.

## Why This Is Different From Traditional Metrics

| External View | Internal View |
|---|---|
| Loss | Activity |
| Accuracy | Contribution (LCS) |
| Precision / Recall | Sensitivity |
| F1 / AUC | Evolution |

Traditional metrics answer *how well is the model performing?* Cerium Delta adds *how is the architecture behaving while it gets there?* — connecting the two views instead of treating them separately.

## Research Philosophy

> **A metric should provide evidence, not mythology.**

A score isn't meaningful just because it produces a number. Every Cerium Delta metric is expected to carry:

- A mathematical definition
- A clear interpretation
- Known assumptions
- Comparable outputs
- Reproducible computation
- Empirical validation
- Stated limitations

The goal is measurements researchers can inspect, question, reproduce, and improve — not a black box that "just knows."

## Framework Support

Cerium Delta is designed to work across ML ecosystems, with model-state extraction kept separate from the analysis layer so the core metrics stay framework-independent:

- PyTorch
- TensorFlow
- JAX
- scikit-learn
- ONNX


Pipeline:

```text
Model extraction → Internal state → Metric computation
→ Statistical analysis → Interpretation → Visualization
```

Visualization is only one consumer of the underlying measurements — this separation is deliberate.

## Installation

```bash
pip install cerium-delta
```

```python
from cerium_delta.metrics.brain import NVS
# or
from cerium_delta.exporters.dev import bridge
```

## Vision

```text
Observe the internal state
        ↓
Measure component behavior (LCS, Sensitivity, Evolution)
        ↓
Compare components
        ↓
Track evolution
        ↓
Identify meaningful changes
        ↓
Understand architectural behavior
```

Instead of asking only *"did the model learn?"*, Cerium Delta helps ask:

> Which components contributed? Which changed or became underutilized?
> Where did internal behavior degrade? How did the architecture evolve — and can we measure it systematically?

---

<div align="center">

**Cerium Delta** — Beyond visualization. Beyond conventional metrics.

</div>
