# Generalized Structured Random Vector Functional Link Network (GS-RVFL)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of **GS-RVFL**, a structured extension of the Random Vector Functional Link network that partitions deterministic inputs into structured direct-link operators and adaptive bias operators.

## Key Features

- **Structured Architecture**: Distinguishes between variables based on their functional role
- **Closed-Form Learning**: Maintains ridge-regression-based output learning
- **Universal Approximation**: Preserves theoretical guarantees of RVFL
- **Domain-Specific Operators**: Pre-built operators for motion, sensor, and skeleton data
- **Comprehensive Evaluation**: 25 benchmark datasets across 4 domains

## Installation

```bash
pip install -r requirements.txt
python setup.py install