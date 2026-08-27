# Contributing to Cerium-Delta

All contributions via Pull Request. No direct push to `main`.

## How to Contribute

1. Fork -> Branch `feat/your-metric` -> PR to `main`
2. One feature per PR (e.g., one metric: LCS, IFS, AHI)
3. For new metrics, include: math definition, assumptions, limitations, test
4. CI must pass

## Core Community Member

### BOOTSTRAP PHASE - First 10 Core Members [ACTIVE NOW]

Since this is early research + engineering, first 10 core members need only:

> **1 Quality PR to become Core**

Quality PR means:
- New observability metric (LCS, Sensitivity, Evolution, etc.)
- Framework exporter (PyTorch, TF, JAX, ONNX)
- Architecture health detector (vanishing gradient, dead neuron, bottleneck)
- Real-time tracking improvement or visualization
- Not typo/formatting/AI spam

Final invite during bootstrap stays with @NEURAL-Y.

Ends after 10 core members.

### After Bootstrap - Full Rules

Meet ONE:

**A) 1 Major Impact**
- New core system: Neuron Vitality Benchmark (NVB), Information Flow Score, Architecture Health Index
- Major framework support or real-time evolution tracking engine

**B) 10 Middle-Level PRs**
- New layer analysis, evolution tracker, health indicator, exporter improvement

**C) 15 Small Improvements**
- Bug fixes, tests, docs, metric validation
- Spam rule: typo/formatting = 0.5 PR

### Core Benefits

- Review other PRs
- Direct commit to non-main branches (`dev/*`, `research/*`, `exp/*`) with your name
- Refer other researchers/devs to core
- Merge authority
- **Governance: Can change CONTRIBUTING.md and PR rules by voting**

### Governance - Voting

- Any core can propose change to CONTRIBUTING.md / PR rules via Proposal PR
- Vote period: 7 days
- Pass: >50% YES + owner approval if core < 10, >60% YES if core >= 10
- Each core = 1 vote
- Owner veto during bootstrap

### Merge Policy

- `main` protected, PR only
- If total core < 3: 1 core approval to merge
- If total core >= 3: 2 core approvals to merge
- Core can refer another core for second review
- No force push

### Branch Flow
