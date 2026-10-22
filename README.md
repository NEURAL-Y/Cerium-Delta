<div align="center">

<img src="https://raw.githubusercontent.com/NEURAL-Y/cerium-delta/main/public/Cerium-delta.png" alt="Cerium Delta" width="360"/>

# Cerium Delta

**Beyond visualization. Toward measurable neural architecture behavior.**

[![License](https://img.shields.io/badge/License-Apache_2.0-cyan.svg)](https://github.com/NEURAL-Y/cerium-delta/blob/main/LICENSE)
[![release](https://img.shields.io/badge/release-v1.1.3-orange)](https://github.com/NEURAL-Y/cerium-delta/releases/tag/v1.1.3)
[![status](https://raw.githubusercontent.com/NEURAL-Y/cerium-delta/main/public/badge-shimmer.svg)](https://cerium-delta.pages.dev)
![paper](https://raw.githubusercontent.com/NEURAL-Y/cerium-delta/main/public/badge-paper.svg)
[![architecture](https://img.shields.io/badge/architecture-Cerium%20Delta-1f6feb)](https://github.com/NEURAL-Y/cerium-delta/blob/main/docs/ARCHITECTURE.md)
[![Quick Start](https://img.shields.io/badge/Quick%20Start-Cerium%20Delta-1f6feb)](https://github.com/NEURAL-Y/cerium-delta/blob/main/docs/quick_start.md)
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

---

<details>
<summary><b>The Problem</b></summary>

<br>

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

</details>

<details>
<summary><b>A Simple Mental Model</b></summary>

<br>

Think of a neural network as a machine with many moving components. Traditional evaluation looks at the machine's final output. Cerium Delta inspects the machine **while it runs** — neuron activity, layer behavior, parameter change, gradient behavior, information propagation — and turns those observations into measurements comparable across layers, training stages, and models.

For researchers and engineers, the same idea is an **observability layer over neural architectures**.

</details>

<details>
<summary><b>Neural Activity</b></summary>

<br>

Cerium Delta analyzes internal behavior including:

- Neuron activity and inactivity
- Dead / saturated neurons
- Activation distributions
- Gradient propagation
- Weight evolution
- Dropout behavior
- Layer utilization

These are the lower-level observations from which higher-level analysis is built.

</details>

<details>
<summary><b>Architecture-Level Analysis</b></summary>

<br>

Individual values are rarely enough on their own. Cerium Delta analyzes how components behave **relative to the rest of the architecture** — moving from *"this tensor changed"* to *"this component changed significantly relative to the rest of the architecture."* That distinction is central to the project.

</details>

<details>
<summary><b>From Measurements to Meaning</b></summary>

<br>

Cerium Delta's current metrics translate internal model state into comparable measures of neural behavior:

| Metric | Purpose |
|---|---|
| **Layer Contribution Score (LCS)** | Measures a layer's relative contribution to the network — not whether it holds large weights, but whether it exhibits meaningful contribution relative to other layers. |
| **Sensitivity** | Measures how strongly a quantity responds to changes in its local or parameterized representation — a perspective distinct from raw magnitude. |
| **Evolution** | Measures how model state changes relative to a reference state across training, turning *"what is the model now?"* into *"how did the model get here?"* |

<details>
<summary><b>Layer Contribution Score (LCS)</b></summary>

<br>

Investigates whether a layer contributes meaningfully to the network, using a relative rather than absolute formulation — large weights alone don't imply importance.

</details>

<details>
<summary><b>Sensitivity</b></summary>

<br>

Measures the responsiveness of a quantity to local or parameter-level perturbation, independent of its raw magnitude.

</details>

<details>
<summary><b>Evolution</b></summary>

<br>

Tracks state change relative to a reference point across training, making the learning trajectory itself observable rather than only the endpoint.

</details>

</details>

<details>
<summary><b>Relative Measurement Matters</b></summary>

<br>

A large layer naturally has more parameters than a small one — a large raw value doesn't automatically mean a component is more important. Cerium Delta leans on **relative analysis**, including percentile-based comparisons and statistical filtering, so components are judged against an appropriate reference population rather than in isolation.

**Ranking provides context; raw measurements preserve precision.**

</details>

<details>
<summary><b>Neural Vitality</b></summary>

<br>

A central research direction: is a neural component actively participating in learning, and how does that change over time? LCS, Sensitivity, and Evolution each offer a different lens on the same underlying question — no single metric is treated as a complete verdict; together they form a measurement system rather than a single score.

</details>

<details>
<summary><b>⚠️ Important: How to Read the Metrics</b></summary>

<br>

Read these first. They prevent the most common misreadings.

- **Values are relative, not absolute.** A score describes a component *compared with the rest of the architecture*, not a universal grade. A large layer will naturally show larger raw values than a small one.
- **Ranking gives context; raw values give precision.** Use the ranking to see where a component sits, and the raw measurement when you need the exact magnitude.
- **No single metric is a verdict.** LCS, Sensitivity and Evolution each show a different lens. Look at them together before drawing a conclusion.
- **Evolution depends on its reference state.** It measures change *relative to a reference*, so choose the reference deliberately and keep it the same when comparing runs.
- **Compare like with like.** Compare layers, training stages or models only when the measurement setup is the same.
- **A metric points where to look. It does not explain why.** A surprising value is a prompt to investigate, not a conclusion.

<details>
<summary><b>Common misreadings</b></summary>

<br>

| If you think... | Remember |
|---|---|
| "Large weights mean an important layer." | LCS is relative. Large weights alone don't imply importance. |
| "A big raw value means this component matters more." | Larger components have more parameters, so compare against an appropriate reference population. |
| "One score tells me if the model is healthy." | The metrics form a measurement system, not a single score. |
| "Low accuracy tells me which layer is wrong." | Traditional metrics show the outcome. The internal view shows where behavior changed. |
| "A low value proves a component is useless." | Treat it as evidence to examine, together with the other metrics. |

</details>

</details>

<details>
<summary><b>Quick Glossary</b></summary>

<br>

| Term | Meaning |
|---|---|
| **Dead neuron** | A neuron that stays inactive, so it contributes nothing to the output. |
| **Saturated neuron** | A neuron pushed into a flat region of its activation function, where it barely responds to changes. |
| **Activation distribution** | How a layer's output values are spread across neurons and inputs. |
| **Gradient propagation** | How the learning signal travels backward through the layers during training. |
| **Weight evolution** | How a layer's weights change over the course of training. |
| **Dropout behavior** | How randomly disabling neurons during training affects activity. |
| **Layer utilization** | How much of a layer's capacity is actually being used. |
| **Reference state** | The earlier model state that Evolution is measured against. |
| **Relative analysis** | Judging a component against other components rather than in isolation. |

</details>

<details>
<summary><b>Why This Is Different From Traditional Metrics</b></summary>

<br>

| External View | Internal View |
|---|---|
| Loss | Activity |
| Accuracy | Contribution (LCS) |
| Precision / Recall | Sensitivity |
| F1 / AUC | Evolution |

Traditional metrics answer *how well is the model performing?* Cerium Delta adds *how is the architecture behaving while it gets there?* — connecting the two views instead of treating them separately.

</details>

<details>
<summary><b>Research Philosophy</b></summary>

<br>

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

</details>

<details>
<summary><b>Framework Support</b></summary>

<br>

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

</details>

<details open>
<summary><b>Installation</b></summary>

<br>

```bash
pip install cerium-delta
```

```python
from cerium_delta.metrics.brain import NVS
# or
from cerium_delta.exporters.dev import bridge
```

</details>

<details>
<summary><b>Vision</b></summary>

<br>

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

</details>

---

<div align="center">

**Cerium Delta** — Beyond visualization. Beyond conventional metrics.

</div>
