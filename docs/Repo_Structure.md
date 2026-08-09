# Cerium Delta — Repository Structure

Real-time neural network observability platform  
From model extraction → intelligence → visualization

---

## Project Layout

```text

cerium-delta/
│
├── src/
│   └── cerium_delta/
│       │
│       ├── __init__.py
│       │
│       ├── dev.py
│       │
│       ├── metrics/
│       │   └── brain.py
│       │
│       ├── exporters/
│       │   ├── torch_converter.py
│       │   ├── tensorflow_converter.py
│       │   ├── jax_converter.py
│       │   └── sklearn_converter.py
│       │
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
