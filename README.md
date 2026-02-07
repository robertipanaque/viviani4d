# Viviani4D: Python Framework for 4D NURBS Geometric Modeling

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15514237.svg)](https://doi.org/10.5281/zenodo.15514237)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Viviani4D is an open-source Python framework for exact NURBS representation and interactive visualization of four-dimensional algebraic hypersurfaces. The framework enables researchers and educators to construct, manipulate, and explore 4D geometric objects using industry-standard NURBS representations.

This software accompanies the manuscript: **"Viviani4D: A Python framework for exact NURBS representation and interactive visualization of 4D algebraic hypersurfaces"** submitted to SoftwareX.

## Features

- **Exact NURBS construction** of 4D algebraic hypersurfaces from rational parameterizations
- **Interactive 4D visualization** via orthogonal projections to $\mathbb{R}^3$
- **Surface family generation** from single 4D representations
- **Symbolic validation** of geometric exactness
- **Modular architecture** with reusable B-spline and NURBS components
- **High-quality rendering** with Matplotlib and Plotly

## Installation

### From GitHub
```bash
# Clone the repository
git clone https://github.com/robertipanaque/viviani4d.git
cd viviani4d

# Install dependencies
pip install numpy matplotlib plotly
