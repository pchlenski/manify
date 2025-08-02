# Welcome to Manify 🪐

A Python Library for Learning Non-Euclidean Representations

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/manify.svg)](https://badge.fury.io/py/manify)
[![Tests](https://github.com/pchlenski/manify/actions/workflows/test.yml/badge.svg)](https://github.com/pchlenski/manify/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/pchlenski/manify/branch/main/graph/badge.svg)](https://codecov.io/gh/pchlenski/manify)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-manify.readthedocs.io-blue)](https://manify.readthedocs.io)
[![arXiv](https://img.shields.io/badge/arXiv-2503.09576-b31b1b.svg)](https://arxiv.org/abs/2503.09576)
[![License](https://img.shields.io/github/license/pchlenski/manify)](https://github.com/pchlenski/manify/blob/main/LICENSE)

Manify is a Python library for non-Euclidean representation learning. 
It is built on top of `geoopt` and follows `scikit-learn` API conventions.
The library supports a variety of workflows involving (products of) Riemannian manifolds, including:
- All basic manifold operations (e.g. exponential map, logarithmic map, parallel transport, and distance computations)
- Sampling Gaussian distributions and Gaussian mixtures
- Learning embeddings of data on product manifolds, using features and/or distances
- Training machine learning models on manifold-valued embeddings, including decision trees, random forests, SVMs, 
perceptrons, and neural networks.
- Clustering manifold-valued data using Riemannian fuzzy K-Means
