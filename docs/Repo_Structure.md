# Cerium Delta — Repository Structure

Real-time neural network observability platform  
From model extraction → intelligence → visualization

---

## Project Layout

```text
cerium-delta/
│
├── .github/
│   ├── CODEOWNERS
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/                  # CI/CD
│
├── public/
│   └── Cerium-delta.png            # Project logo
│
├── src/
│   ├── torchvis/                   # PyTorch extractor
│   ├── skvis/                      # scikit-learn extractor
│   ├── tensorvis/                  # TensorFlow extractor
│   ├── dev/                        # Data cleaning & filtering
│   ├── brain/                      # Neuron Vitality System (NVS)
│   └── visualizer/                 # 2D & 3D visualization engine
│
├── tests/                          # Test suite (planned)
│
├── docs/
│   ├── architecture.md
│   └── repo_structure.md
│
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── pyproject.toml           #configuration(pip)
