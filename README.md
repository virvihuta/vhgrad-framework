# VHGrad

**An educational light deep learning framework built step by step from scratch.**
Inspired by Karpathy’s *micrograd*, but expanded into a full journey: **micrograd → micrograd++ → microgradx → vhgrad**.

---

## Project Roadmap

This repository is structured as a staged learning + engineering journey. Each stage builds on the previous one, gradually transforming a toy autograd engine into a portfolio-grade deep learning framework.

### 1. Micrograd (done)

* Similar reproduction of Karpathy’s Micrograd.
* Implements a scalar-based autograd engine with:

  * `Value` class (forward/backward)
  * Core ops: `+`, `*`, `tanh`
  * Computation graph visualization (Graphviz)
  * Minimal `Neuron`, `Layer`, `MLP`

### 2. Micrograd++ (next)

**Goal:** Harden the scalar engine & add missing features.

* Add full op coverage: `sub`, `neg`, `pow`, `/`, `exp`, `log`, `relu`
* Add losses: MSE, Binary Cross Entropy
* Gradient checker (finite differences)
* Optimizers: SGD + momentum
* Utility: `zero_grad()`
* Unit tests with `pytest`

### 3. MicrogradX

**Goal:** Vectorize with NumPy & enable batching.

* `TensorValue`: extend autograd to ndarrays
* Vectorized `Neuron`, `Layer`, `MLP` (matrix multiplies)
* Mini-batching + dataloader
* Optimizers: SGD, Adam
* Benchmarks: ≥5× faster than scalar micrograd++

### 4. VHGrad

**Goal:** Polish into a professional-grade mini-framework.

* Core API:

  * `Module`, `Parameter`, `Sequential`
  * `Linear`, `Dropout`, activations (ReLU, Tanh, LeakyReLU)
* Training utilities:

  * SGD, Adam, AdamW optimizers
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

* XOR: ≥95% acc within 5k steps (micrograd++)
* Spiral: ≥90% acc (microgradx)
* MNIST (1k subset): ≥90% acc ≤10 epochs (microgradx)
* MNIST (10k subset): ≥95% acc ≤20 epochs (vhgrad)
* Speedup: ≥5× vs scalar baseline

---

## Definition of Done per stage

* **micrograd:** scalar engine reproduces tutorial 
* **micrograd++:** ops, losses, grad-checker, XOR demo working
* **microgradx:** tensor engine, batching, spiral & MNIST demos
* **vhgrad:** polished framework, docs, PyPI, ≥95% MNIST acc, CI tests

---

## License

MIT — free to use, learn from, and extend.

---

## Credits

* Original inspiration: \[Andrej Karpathy’s micrograd]
* Built by [Virvi Huta](https://github.com/virvihuta)
