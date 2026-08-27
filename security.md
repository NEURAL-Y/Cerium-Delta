# Security Policy for Cerium-Delta

## Overview
Cerium-Delta is a real-time neural network observability platform. It extracts and analyzes internal model states: activations, gradients, weights, and information flow. Security is critical because it handles model files and numerical data that can lead to code execution if deserialized unsafely.

## Supported Versions

| Version | Supported |
| :--- | :--- |
| main branch | ✅ |
| Latest release | ✅ |
| Pre-release / research badges | ✅ (best-effort) |
| < 0.1.0 | ❌ |

## Reporting a Vulnerability

**DO NOT open a public issue for security vulnerabilities.**

Report privately:

1. **GitHub: Security > Report a vulnerability (Preferred)**
2. **Email:** helloiamnew.main@gmail.com

Include:
- Description + impact
- Repro: model type (PyTorch / TF / JAX / ONNX), Cerium-Delta version, python version
- Sample script or model that triggers it (if possible, dummy weights)
- Whether it requires untrusted model file

### Cerium-Delta Specific Scope

**In Scope (high priority):**
- Arbitrary code execution via model loading (pickle, torch.load, TF SavedModel)
- Path traversal in exporters / model-state extraction
- ReDoS / OOM crash via crafted activation tensors / gradients
- Data exfiltration from observability server / dashboard
- Prototype pollution / injection if JS visualization layer exists
- Supply chain: dependencies that allow code exec during install

**Out of Scope:**
- Model accuracy / metric correctness (that's a bug, not security, unless it leaks data)
- DoS via training huge models (expected resource usage)
- Vulnerabilities in PyTorch / TensorFlow themselves (report upstream)

> WARNING: Never load untrusted models with `torch.load` without `weights_only=True`. Cerium-Delta analyzers must treat all model files as untrusted by default.

## Response Process

1. Acknowledge in 48h
2. Triage + reproduce in 5 days
3. Fix on private branch
4. Release + GHSA Advisory + CVE if needed

We follow 90-day coordinated disclosure.

## Safe Harbor

Good-faith research is allowed. Do not access other users' models/data, do not degrade demo servers.

## Security Best Practices for Users

- Run Cerium-Delta in isolated env when analyzing untrusted models
- Do not expose real-time observability dashboard publicly without auth
- Pin versions, enable Dependabot
- Treat activation dumps as sensitive - they can leak training data

## Updates

Security fixes via GitHub Releases and Security Advisories.

Maintained by NEURAL-Y
