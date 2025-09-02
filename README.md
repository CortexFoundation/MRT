# MRT (Model Representation Template)

## Project Overview

MRT is a Python-based framework for machine learning model quantization and compilation. It is built on top of CVMRuntime and is designed to convert pre-trained models into CVM-Compatiable formats, including fixed-point representations and zero-knowledge (ZK) circuits for verifiable inference.

The core of the framework is the `Trace` object, which represents the computational graph of a model and provides a high-level API for applying various transformations, such as quantization, calibration, and operator fusion.

## Key Features:

*   **Model Ingestion:** Imports models from PyTorch or TVM Relay.
*   **Quantization:** Supports post-training quantization with calibration.
*   **Export:** Exports models to various formats, including:
    *   Simulated quantized formats for accuracy evaluation.
    *   Fixed-point format for blockchain runtime deployment.
*   **Zero-Knowledge Integration:** Generates Circom circuits from quantized models for use in ZK-SNARK-based verifiable inference.

## Architecture:

*   **Frontend:** The `mrt.frontend` module handles the import of models from other frameworks (e.g., PyTorch) into the MRT representation.
*   **MIR (Model Intermediate Representation):** The `mrt.mir` module defines the core data structures for representing the model's computational graph.
*   **Quantization:** The `mrt.quantization` module contains the logic for model quantization, including calibration, scaling, and precision revision.
*   **Runtime:** The `mrt.runtime` module provides tools for model evaluation and analysis.
*   **ZKML:** The `mrt.frontend.zkml` and `mrt.trace_to_circom` modules handle the conversion of models into Circom circuits.

## Key Concepts

*   **Trace:** The central abstraction in MRT. A `Trace` object represents the model's computational graph and provides methods for applying transformations.
*   **Quantization:** The process of converting a model's weights and activations to a lower-precision format (e.g., 8-bit integers). This is essential for deploying models on resource-constrained hardware and for use in ZK-SNARKs.
*   **Calibration:** The process of determining the appropriate scaling factors for quantization. This is typically done by running the model on a small, representative dataset.
*   **Circom:** A domain-specific language for writing arithmetic circuits for ZK-SNARKs. MRT can automatically generate Circom code from a quantized model, enabling verifiable inference.

## Building and Running

### Setup

To set up the Python environment, run the following command from the project root:

```bash
source env.sh
```

This script will add the `python` directory to your `PYTHONPATH`, allowing you to import the `mrt` package.

### Running Tests

Run individual tests:

```bash
# Frontend tests
python tests/frontend/test_frontend_loading.py
python tests/frontend/test.pytorch.py

# Classification model tests  
python tests/classification/test.resnet.py
python tests/classification/test.mnist.py

# TVM/Relax tests
python tests/test.relax.py

# Template for new tests
python tests/test.template.py

# PyTest
pytest tests/frontend/test_pytorch.py::test_conv_model -v
```

All test files should be located in the `tests/` directory with subdirectories for different categories (frontend, classification, detection, nlp).

## Development Conventions

### Git

Git commit message should be simple, and add core feature name at the begin, examples:
    `[python]: add torch module`
    `[tests]: fix test_frontend_loading`

### Code

*   IMPORTANT!!! code should be simply, reduce duplicated codes in programming.
*   IMPORTANT!!! Don't consider For simplicity situations, write code in robust, complete, correct version.
*   write code less than 50 lines once a time, if neccesary.
*   for large changes, split into many segements and functions, and require for user confirmation.

*   The project follows standard Python coding conventions (PEP 8).
*   Print essencial info and exit immediately if any unsupported operation/corner case is performed.
*   use assert to check most scenarios.
*   Add PYTHONPATH with python dir in script.
*   Remove unvisable char(whitespace etc) in empty line.

### Testing

*   Tests are located in the `tests` directory, with subdirectories for different model types and framework components.
*   The tests demonstrate the intended usage of the framework and provide a good starting point for understanding its capabilities.
*   The `tests/test.trace.py` file is particularly important, as it shows the end-to-end workflow from model import to Circom generation.
