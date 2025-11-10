# VHGrad

**An educational light deep learning framework built step by step from scratch.**
Inspired by Karpathy’s *micrograd*, but expanded into a full journey.

---

## Project Roadmap

This repository is structured as a staged learning + engineering journey. Each stage builds on the previous one, gradually transforming a toy autograd engine into a portfolio-grade deep learning framework.

### 1. Micrograd (done)

* Similar reproduction of Karpathy’s Micrograd with some other differences.
* Implements a scalar-based autograd engine with:

  * `Value` class (forward/backward)
  * Core ops: `+`, `-`, `*`, `/`, `^`, `tanh`
  * Computation graph visualization (Graphviz)
  * Minimal `Neuron`, `Layer`, `MLP`

### 4. VHgrad (not released)

**Goal:** Polish into a professional-grade mini-framework.

* Core API:

  * `Module`, `Parameter`, `Sequential`
  * `Linear`, `Dropout`, activations (ReLU, Tanh, LeakyReLU)
* Training utilities:

  * SGD, Adam optimizers
  * LR schedulers (step, cosine)
  * Gradient clipping, early stopping
* Reliability:

  * Save/load checkpoints
  * Deterministic seeds
* Metrics: accuracy, precision/recall, confusion matrix
* Docs site (mkdocs)
* Examples: XOR, Spiral, MNIST (≥95% acc), CIFAR-10 tiny baseline
* Packaged on PyPI (`pip install vhgrad`)
* CI tests (GitHub Actions)

---

## Quickstart

```bash
pip install vhgrad  # after vhgrad stage is released
```

```python
from vhgrad import MLP, SGD, mse

# simple 2-3-1 MLP
model = MLP(2, [3, 1])
opt = SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    ypred = [model(x) for x in xs]
    loss = mse(ys, ypred)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

---

## Benchmarks (targets)

* XOR: ≥95% acc within 5k steps
* Spiral: ≥90% acc
* MNIST (1k subset): ≥90% acc ≤10 epochs
* MNIST (10k subset): ≥95% acc ≤20 epochs
* Speedup: ≥5× vs scalar baseline

---

## Definition of Done per stage

* **micrograd:** scalar engine reproduces tutorial 
* **vhgrad:** polished framework, docs, PyPI, ≥95% MNIST acc, CI tests

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE) — free to use, learn from, and extend.


---

## Credits

* Original inspiration: [Andrej Karpathy’s micrograd](https://github.com/karpathy/micrograd)
* Built by [Virvi Huta](https://github.com/virvihuta)
