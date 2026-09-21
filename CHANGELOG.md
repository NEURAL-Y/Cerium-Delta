# Changelog

All notable changes to **Cerium Delta** are documented here.

## [1.1.0] - 2026-09-21

### Added

* Introduced **statistical analysis and static visualization** for Neural Vitality data.
* Added `visualizer` for analytical plots using Matplotlib and Seaborn.
* Added `Statistical` for numerical and statistical analysis.
* Added support for:

  * Bar, box, histogram, IQR, and MAD analysis
  * Linear regression and fitting
  * Correlation and covariance analysis
  * Large-scale distribution visualization with Datashader
* Visualization and analysis support for **LCS, Sensitivity, and Evolution** across weights and biases.
* Expanded public API:

```python
from cerium_delta import NVS, bridge, visualizer, Statistical
```

* Added **ONNX** integration.

### Dependencies

Added visualization and analysis support through **Pandas, Matplotlib, Seaborn, and Datashader**, alongside the existing NumPy and SciPy stack.

### Notes

v1.1.0 expands Cerium Delta from core Neural Vitality metrics into a broader **observability and analytical inspection layer**.

**Full Changelog:** https://github.com/NEURAL-Y/Cerium-Delta/compare/v1.0.1...v1.1.0

---

## [1.0.1] - 2026-08-25

### Changed

* Refactored **Bridge and NVS return types** for a cleaner, more consistent API.
* Improved framework handling and `compute_choice` behavior.
* Standardized error and framework-related result handling.
* Cleaned up API inconsistencies and typos.

### Added

* Added **ONNX Runtime tensor interoperability**.

### Notes

Patch release focused on **API consistency, stability, and interoperability**.

```bash
pip install --upgrade cerium-delta
```

---

## [1.0.0] - 2026-08-21

### Added

* Initial stable release of **Cerium Delta**.
* **Layer Contribution Score (LCS)**.
* **Sensitivity analysis**.
* **Evolution analysis**.
* Weight and bias analysis.
* Core neural observability infrastructure.

### Notes

The initial foundation for **Neural Vitality analysis and neural network observability**.
