## Complete PyTorch Example

The following example creates a small PyTorch model, trains it, and then analyzes it with Cerium Delta v1.0.1

### Install dependencies

```bash
pip install cerium-delta torch
```

**Output**

`Cerium Delta and PyTorch are installed.`

### Create and train a model

```python
import torch
import torch.nn as nn
import torch.optim as optim

from cerium_delta.dev import bridge


# Create a simple neural network
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)


# Example training data
x = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [2.0]
])


# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Train the model
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()

    prediction = model(x)
    loss = criterion(prediction, y)

    loss.backward()
    optimizer.step()


# Create the Cerium Delta bridge
analyzer = bridge(
    model,
    framework="torch",
    compute_choice="lcs"
)


# Run NVS analysis
result = analyzer.nvs_export_info()

print(result)
```

**Output**

`{'layer_contribution_score': {'weights': {...}, 'biases': {...}}}`

The exact score values depend on the trained model parameters.

---

## Analyze Different NVS Metrics

The same trained model can be analyzed using different computation modes.

### Layer Contribution Score

```python
analyzer = bridge(
    model,
    framework="torch",
    compute_choice="lcs"
)

result = analyzer.nvs_export_info()

print(result)
```

**Output**

`{'layer_contribution_score': {'weights': {...}, 'biases': {...}}}`

### Sensitivity

```python
analyzer = bridge(
    model,
    framework="torch",
    compute_choice="sensitivity"
)

result = analyzer.nvs_export_info()

print(result)
```

**Output**

`{'sensitivity': {...}}`

### Evolution

```python
analyzer = bridge(
    model,
    framework="torch",
    compute_choice="evolution"
)

result = analyzer.nvs_export_info()

print(result)
```

**Output**

`{'evolution': {...}}`

### All Available Metrics

```python
analyzer = bridge(
    model,
    framework="torch",
    compute_choice="all"
)

result = analyzer.nvs_export_info()

print(result)
```

**Output**

`{'layer_contribution_score': {...}, 'sensitivity': {...}, 'evolution': {...}}`

The exact returned structure depends on the selected computation mode.

---

# Complete Minimal Workflow

If you want the shortest practical PyTorch workflow, use:

```python
import torch
import torch.nn as nn

from cerium_delta.dev import bridge


# Create a model
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)


# Analyze the model
analyzer = bridge(
    model,
    framework="torch",
    compute_choice="lcs"
)

result = analyzer.nvs_export_info()

print(result)
```

**Output**

`{'layer_contribution_score': {'weights': {...}, 'biases': {...}}}`

This workflow analyzes the model parameters directly. Training the model first is recommended when you want meaningful parameter evolution and trained-state analysis.

---
# ONNX Array Export

Cerium Delta also provides `pyarr_to_onnx()` for converting a NumPy array into an ONNX file containing that array.

## Install ONNX

ONNX is optional in Cerium Delta.

If you want to use the ONNX exporter, install it separately:

```bash
pip install onnx
# Quick Reference

### Install Cerium Delta and PyTorch

```bash
pip install cerium-delta torch
```

**Output**

`Cerium Delta and PyTorch are installed.`

### Import Bridge

```python
from cerium_delta.dev import bridge
```

**Output**

`bridge`

### Create a bridge analyzer

```python
analyzer = bridge(
    model,
    framework="torch",
    compute_choice="lcs"
)
```

**Output**

`analyzer`

### Run the analysis

```python
result = analyzer.nvs_export_info()
```

**Output**

`result`

### Access weight scores

```python
result["layer_contribution_score"]["weights"]
```

**Output**

`{"layer 0": 0.82, "layer 1": 0.64, ...}`

### Access bias scores

```python
result["layer_contribution_score"]["biases"]
```

**Output**

`{"layer 0": 0.31, "layer 1": 0.47, ...}`

---

## Next Step

After getting the NVS result, the returned dictionary can be passed to your own analysis or visualization pipeline to investigate how different layers contribute, respond, and evolve.

