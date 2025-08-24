---

# MicrogradX

**MicrogradX** is a lightweight deep learning framework in Python, 
inspired by Karpathy’s [micrograd](https://github.com/karpathy/micrograd).
It’s built entirely from scratch to demonstrate how neural networks and 
automatic differentiation work under the hood.

This project is organized into **three stages of progression**:

---

## 📌 Project Roadmap

### 1. **Micrograd (Stage 1)**

A faithful reimplementation of Karpathy’s *micrograd*:

* Core autograd engine (computational graph)
* Basic forward & backward propagation
* Elementary scalar operations

👉 Goal: understand **how automatic differentiation works** from scratch.

---

### 2. **Micrograd++ (Stage 2)**

My own extension of *micrograd* with improvements:

* Support for vectors, matrices, and batched operations
* Additional activation functions (ReLU, Sigmoid, Tanh, etc.)
* Better visualization of computational graphs

👉 Goal: expand *micrograd* into something closer to a **mini deep 
learning library**.

---

### 3. **MicrogradX (Stage 3)**

The advanced version of the project:

* Implementation of a small neural network library (MLPs, etc.)
* Training with gradient descent and optimizers
* Example projects (e.g. classification tasks)
* Clear math explanations alongside the code

👉 Goal: build a **lightweight deep learning framework** that bridges the 
gap between educational code and real-world PyTorch/TensorFlow.

---

## ✨ Features (current & planned)

* [x] Scalar autograd engine
* [x] Backpropagation on simple computational graphs
* [ ] Vector/matrix support
* [ ] Common activation functions (ReLU, Sigmoid, Tanh)
* [ ] Multilayer Perceptrons (MLPs)
* [ ] Training on toy datasets (XOR, MNIST subset, etc.)

---

## 🚀 Why this project?

Most frameworks hide the math. **MicrogradX** forces me to build it step 
by step, so I (and others) can:

* See how backpropagation really works
* Connect calculus (chain rule, Jacobians) with code
* Create a foundation before diving into PyTorch/TensorFlow

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/virvihuta/MicrogradX.git
cd MicrogradX
```

---

## 📚 Examples (coming soon)

* Train a simple MLP on XOR
* Handwritten digit recognition (MNIST subset)
* Visualize computational graphs

---

## 📖 Credits

* Original idea from [Andrej Karpathy’s 
micrograd](https://github.com/karpathy/micrograd)
* Extended and reimagined as part of my deep learning learning journey

---

