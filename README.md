# Homework 3 – JAX and PyTorch Compilation

This repository contains the implementation and report for Homework 3.

## Files

hw3.pdf  
Final report including experiment results and analysis.

jax_experiments.ipynb  
Notebook containing JAX experiments including JIT compilation overhead and shape specialization.

torch_compile_analysis.ipynb  
Notebook containing PyTorch compile backend comparison, debugging compilation failures, and graph capture analysis.

figures/  
Contains plots generated during experiments.

## Experiments

### JAX Experiments
- JIT compilation overhead
- Shape specialization

### PyTorch Experiments
- Backend performance comparison
- Debugging compilation failures
- Graph capture and FX graph inspection

## Summary

The experiments demonstrate how compilation techniques improve performance in modern machine learning frameworks. JIT compilation allows optimized execution after the initial compile step, while PyTorch compilation backends reduce Python overhead and enable better kernel optimization.
