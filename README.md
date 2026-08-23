# CERIUM-DELTA
![CERIUM-DELTA](./public/Cerium-delta.png "CERIUM-DELTA-LOGO")


# Beyond Visualization
![SAMPLE](./public/sample.png.png "sample image")<br>
Cerium Delta is not a neural network drawing tool.
Most architecture visualization tools generate static diagrams that describe how a model is constructed. While useful for documentation, they provide little insight into how a model behaves during training or inference.

Cerium Delta focuses on **observability**, not just visualization.

The objective is to provide real-time insight into the internal dynamics of neural architectures, allowing researchers and developers to understand how information flows through a model as it learns.<br>
# Build stack
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![turtle](https://img.shields.io/badge/turtle-Python%20Stdlib-blue?style=for-the-badge&logo=python&logoColor=white)](https://docs.python.org/3/library/turtle.html)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-visualizations-blue?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![JAX](https://img.shields.io/badge/JAX-autograd%20%26%20XLA-orange?style=for-the-badge&logo=python&logoColor=white)](https://jax.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-deep%20learning-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-deep%20learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![NumPy](https://img.shields.io/badge/NumPy-numerical%20computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![joblib](https://img.shields.io/badge/joblib-parallel%20computing-yellowgreen?style=for-the-badge&logo=python&logoColor=white)](https://joblib.readthedocs.io/)
[![SciPy](https://img.shields.io/badge/SciPy-scientific%20computing-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-machine%20learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
---
# Other Configure
[![License](https://img.shields.io/badge/License-Apache_2.0-cyan.svg)](https://github.com/NEURAL-Y/cerium-delta/blob/main/LICENSE)
[![release](https://img.shields.io/badge/release-v1.0.0-orange)](https://github.com/NEURAL-Y/cerium-delta/tags)
[![status](https://raw.githubusercontent.com/NEURAL-Y/cerium-delta/main/public/badge-shimmer.svg)](https://cerium-delta.pages.dev)
![paper](https://raw.githubusercontent.com/NEURAL-Y/cerium-delta/main/public/badge-paper.svg)
<a href="https://github.com/NEURAL-Y/cerium-delta/blob/main/docs/Cerium_Delta_Development_Charter.md"><img src="https://raw.githubusercontent.com/NEURAL-Y/cerium-delta/main/public/cerium_delta_charter_badge.svg" alt="Cerium Delta - Development Charter" /></a>
[![architecture](https://img.shields.io/badge/architecture-Cerium%20Delta-1f6feb)](https://github.com/NEURAL-Y/cerium-delta/blob/main/docs/ARCHITECTURE.md)
# What We Observe

Cerium Delta aims to monitor the internal state of neural architectures in real time.

# The Problem

A neural network is usually evaluated from the outside.

We ask:

```text
Did the loss decrease?
Did accuracy improve?
Did the model generalize?
```

These questions are important, but they describe the **outcome** of learning.

They do not completely describe the **process**.

When something goes wrong, a developer or researcher may still have to ask:

```text
Which layer changed?

Which layer stopped contributing?

Where did the signal weaken?

Which parameters evolved significantly?

Which neurons became inactive?

Is a layer being used effectively?

Did the architecture learn a useful representation?

Did the model become more stable or less stable during training?
```

Cerium Delta is intended to provide measurements that help investigate these questions.

---

# What Cerium Delta Is

Cerium Delta is **not primarily a neural-network drawing tool**.

Architecture visualization answers:

> What does the model look like?

Cerium Delta focuses on:

> **How is the model behaving?**

It combines model-state extraction, numerical analysis, architecture-aware metrics, and visualization to study neural networks during training and analysis.

The long-term objective is to make neural architectures **observable, measurable, and analyzable as dynamic systems**.

---

# A Simple Mental Model

For beginners, the idea can be understood like this:

Imagine a neural network as a large machine containing many components.

Traditional evaluation mainly looks at the final output of the machine.

Cerium Delta tries to inspect the machine while it is operating.

It observes things such as:

```text
Neuron activity
Layer behavior
Parameter changes
Gradient behavior
Information propagation
Layer contribution
Architecture evolution
```

It then converts these observations into measurements that can be compared across layers, neurons, training stages, and models.

For researchers and professional developers, the same idea can be viewed as an **observability layer over neural architectures**.

---

## Neural Activity

Cerium Delta can analyze internal neural behavior such as:

* Neuron activity
* Neuron inactivity
* Dead neurons
* Saturated neurons
* Activation distributions
* Gradient propagation
* Weight evolution
* Dropout behavior
* Layer utilization

These signals provide the lower-level observations from which higher-level analysis can be constructed.

---

# Architecture-Level Analysis

Individual values are not always enough.

Cerium Delta therefore aims to analyze how components behave relative to the rest of the architecture.

Examples include:

* Layer contribution
* Layer utilization
* Parameter evolution
* Information propagation
* Neuron importance
* Structural changes
* Representation behavior
* Architecture evolution

This allows the system to move from:

> "This tensor changed."

toward:

> **"This component changed significantly relative to the rest of the architecture."**

That distinction is central to Cerium Delta.

---

# From Measurements to Meaning

Cerium Delta is not intended to stop at collecting numbers.

The research direction is to develop metrics that transform internal model states into **comparable measurements of neural behavior**.

For example:

### Layer Contribution Score (LCS)

Measures the relative contribution of layers according to the mathematical formulation being developed by Cerium Delta.

The purpose is not simply to say that a layer has large weights.

The purpose is to investigate whether a layer exhibits meaningful contribution relative to other layers.

### Sensitivity

Measures how strongly the analyzed quantity responds to changes in its local or parameterized representation.

This provides another perspective that is different from simply measuring magnitude.

### Evolution

Measures how the model state changes relative to a reference state across training.

This makes it possible to study not only:

> "What is the model now?"

but also:

> **"How did the model get here?"**

---

# Relative Measurement Matters

Raw values can be misleading when comparing different layers.

A large layer naturally contains more parameters than a small layer.

A large numerical value does not automatically mean that a component is more important.

Cerium Delta therefore explores **relative analysis**, including percentile-based comparisons and statistical filtering.

The objective is to compare components within an appropriate reference population rather than interpreting isolated numbers as absolute truth.

This is especially important when multiple layers produce similar measurements.

For example, if two layers receive similar percentile scores, their underlying measurements can still be inspected to understand how close they actually are.

In other words:

**ranking provides context; raw measurements preserve precision.**

---

# Neural Vitality

One of the central research directions of Cerium Delta is the idea of measuring the **vitality of neural components**.

The basic question is:

> Is a neural component actively participating in the learning process, and how does its behavior change over time?

This leads toward measurements involving:

* Activity
* Contribution
* Sensitivity
* Evolution
* Utilization
* Relative importance

The goal is not to declare that a single metric can perfectly determine whether a neuron or layer is "useful."

Instead, multiple signals can provide different perspectives on the same component.

This makes neural analysis more like a measurement system than a single score.

---

# Neuron Vitality Benchmark

Cerium Delta also aims toward a broader benchmarking system for evaluating neural architectures.

The **Neuron Vitality Benchmark (NVB)** is intended to provide a standardized environment for comparing internal neural behavior across models and architectures.

Rather than comparing models only through:

```text
Accuracy
Loss
F1
AUC
```

the benchmark can investigate additional dimensions such as:

```text
Layer contribution
Sensitivity
Parameter evolution
Neuron vitality
Layer utilization
Architectural stability
Internal behavioral changes
```

The purpose is not to replace conventional benchmarks.

It is to provide another dimension of evaluation:

> **How does the architecture behave internally while achieving its external performance?**

---

# Architecture Intelligence

The long-term goal is to move from **observation** toward **architecture intelligence**.

A useful observability system should eventually be capable of turning measurements into interpretable findings.

For example:

```text
Layer 12 contribution decreased significantly.

Attention Block 6 shows abnormal gradient behavior.

Feed Forward Layer 4 appears underutilized.

Activation density is decreasing during training.

Information propagation degraded between Layers 8 and 9.

A group of neurons shows sustained inactivity.

Representation diversity decreased during later training.

A substantial architectural change occurred after epoch 40.
```

These statements are not intended to replace the underlying measurements.

They are interpretations built on top of them.

The underlying data should remain available so that researchers can inspect and validate the conclusion.

---

# Why This Is Different From Traditional Metrics

Traditional training metrics primarily answer:

```text
How well is the model performing?
```

Cerium Delta asks additional questions:

```text
How is the architecture behaving?

Which components are changing?

How are components contributing relative to each other?

Where is behavior becoming unstable?

Which parts of the architecture are being utilized?

How does the internal state evolve during learning?
```

This creates two complementary views of a model.

### External View

```text
Loss
Accuracy
Precision
Recall
F1
AUC
```

### Internal View

```text
Activity
Contribution
Sensitivity
Evolution
Utilization
Gradient behavior
Information propagation
Architecture dynamics
```

Cerium Delta focuses on connecting these two views.

---

# Real-Time Architecture Evolution

Neural networks are not static objects during training.

Their parameters continuously change.

Representations change.

Gradients change.

Neuron behavior changes.

Layer behavior changes.

Therefore, inspecting only the final model state can hide important parts of the learning process.

Cerium Delta aims to track these changes over time.

## Evolution Tracking

* Neuron behavior over time
* Layer contribution evolution
* Parameter evolution
* Information-flow evolution
* Gradient trends
* Architecture health evolution
* Learning-phase transitions
* Changes in layer utilization

The objective is to make training a **time-dependent observable process**, rather than a sequence of isolated checkpoints.

---

# Architecture Health

Cerium Delta also explores higher-level indicators describing the state of an architecture.

Potential areas include:

* Bottleneck detection
* Vanishing-gradient detection
* Exploding-gradient detection
* Representation-collapse detection
* Underutilized-layer detection
* Redundant-layer detection
* Structural efficiency
* Learning stability

These signals should be treated as analytical indicators rather than absolute diagnoses.

A metric can indicate that something deserves investigation.

It should not pretend to magically understand a neural network.

---

# Research Metrics

Cerium Delta is developing architecture-aware metrics beyond conventional training statistics.

| Metric                                   | Purpose                                                  |
| ---------------------------------------- | -------------------------------------------------------- |
| **Layer Contribution Score (LCS)**       | Analyze relative layer contribution                      |
| **Sensitivity**                          | Measure response behavior of model quantities            |
| **Evolution**                            | Measure change relative to a reference state             |
| **Neuron Vitality**                      | Analyze neural component activity and behavior           |
| **Information Flow Score (IFS)**         | Analyze information propagation through the architecture |
| **Architecture Health Index (AHI)**      | Aggregate indicators of architectural condition          |
| **Structural Efficiency Score (SES)**    | Analyze effective parameter utilization                  |
| **Learning Stability Score (LSS)**       | Analyze stability during learning                        |
| **Representation Diversity Score (RDS)** | Analyze diversity of learned representations             |

These metrics are research-oriented.

Their definitions, assumptions, limitations, and validation should be documented as the project develops.

---

# Research Philosophy

Cerium Delta follows a simple principle:

> **A metric should provide evidence, not mythology.**

A score should not be treated as meaningful merely because it produces a number.

Each metric should ideally have:

* A mathematical definition
* A clear interpretation
* Known assumptions
* Comparable outputs
* Reproducible computation
* Empirical validation
* Appropriate limitations

The goal is to build measurements that researchers can inspect, question, reproduce, and improve.

---

# Framework Support

Cerium Delta is designed to operate across different machine-learning ecosystems.

Current and planned support includes:

* PyTorch
* TensorFlow
* JAX
* scikit-learn
* ONNX

The framework-specific exporters/converters separate model-state extraction from the analysis layer.

This allows Cerium Delta's analysis concepts to remain as framework-independent as possible.

---

# Architecture

```text
cerium-delta/
├── src/
│   └── cerium_delta/
│       ├── __init__.py
│       ├── dev.py
│       ├── metrics/
│       │   └── brain.py
│       ├── exporters/
│       │   ├── torch_converter.py
│       │   ├── tensorflow_converter.py
│       │   ├── jax_converter.py
│       │   └── sklearn_converter.py
│       └── visualization/
│           └── viz.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   └── Repo_Structure.md
│
├── tests/
│
├── README.md
├── LICENSE
├── pyproject.toml
└── .gitignore
```

The architecture separates:

```text
Model extraction
       ↓
Internal state
       ↓
Metric computation
       ↓
Statistical analysis
       ↓
Interpretation
       ↓
Visualization
```

This separation is important because visualization is only one possible consumer of the measurements.

---

# Roadmap

## Core Observability

* [ ] Activation Tracking
* [ ] Gradient Tracking
* [ ] Weight Evolution Tracking
* [ ] Dropout Monitoring
* [ ] Information Flow Tracking

## Neural Analysis

* [ ] Layer Contribution Score
* [ ] Sensitivity Analysis
* [ ] Evolution Analysis
* [ ] Neuron Vitality Analysis
* [ ] Layer Utilization Analysis

## Architecture Intelligence

* [ ] Information Flow Analysis
* [ ] Bottleneck Detection
* [ ] Dead Neuron Detection
* [ ] Representation Collapse Detection
* [ ] Layer Redundancy Detection
* [ ] Architecture Health Analysis

## Benchmarking

* [ ] Neural Vitality Benchmark
* [ ] Cross-architecture comparison
* [ ] Metric validation
* [ ] Reproducible evaluation protocols
* [ ] Research benchmark suite

## Framework Support

* [ ] PyTorch
* [ ] TensorFlow
* [ ] JAX
* [ ] ONNX
* [ ] scikit-learn

## Platform

* [ ] Real-Time Dashboard
* [ ] Architecture Timeline
* [ ] Architecture Comparison Engine
* [ ] Distributed Training Support
* [ ] Research Visualization

---
# How to use
``` pip install cerium-delta ```
# Vision

Modern machine-learning systems are increasingly complex.

As architectures become larger and more dynamic, knowing that a model performs well is not always enough.

We also need to understand **how the architecture behaves while producing that performance**.

Cerium Delta aims to build the measurement layer for that problem.

The vision is to move neural-network analysis from:

```text
Observe the output
        ↓
Measure performance
        ↓
Accept or reject the model
```

toward:

```text
Observe the internal state
        ↓
Measure component behavior
        ↓
Compare components
        ↓
Track evolution
        ↓
Identify meaningful changes
        ↓
Understand architectural behavior
```

The ultimate goal is not to claim that a single metric can explain a neural network.

The goal is to build a system in which **many measurable signals can be combined to investigate why a neural architecture behaves the way it does.**

Instead of asking only:

> **"Did the model learn?"**

Cerium Delta aims to help ask:

> **"What changed while it learned?"**

> **"Which components contributed?"**

> **"Which components changed or became underutilized?"**

> **"Where did internal behavior degrade?"**

> **"How did the architecture evolve?"**

> **"Can we measure these changes systematically?"**

Cerium Delta is an effort toward making neural networks **observable, measurable, and scientifically analyzable as dynamic systems.**

---

# CERIUM-DELTA

**Beyond visualization.**
**Beyond conventional metrics.**
**Toward measurable neural architecture behavior.**


> How did the architecture evolve during learning?

Cerium Delta is an effort toward making neural networks observable, explainable, and measurable as dynamic systems.
