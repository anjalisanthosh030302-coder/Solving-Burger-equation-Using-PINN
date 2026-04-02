# Solving-Burger-equation-Using-PINN
# 🔬 Physics-Informed Neural Network (PINN) for Burgers’ Equation

This project implements a **Physics-Informed Neural Network (PINN)** using PyTorch to solve the **1D Burgers’ Equation**, a nonlinear partial differential equation that exhibits shock wave behavior.

## 📌 Problem Statement

We solve the Burgers’ equation:

∂u/∂t + u ∂u/∂x − (0.01/π) ∂²u/∂x² = 0

### Initial Condition:
u(x,0) = -sin(πx)

### Boundary Conditions:
u(-1,t) = 0  
u(1,t) = 0  

## 🚀 Key Features

- ✅ Physics-Informed Neural Network (PINN)
- ✅ Automatic differentiation for PDE constraints
- ✅ No training dataset required
- ✅ Adam + L-BFGS optimization
- ✅ Shock wave modeling
- ✅ Exact solution comparison
- ✅ Error evaluation (Relative L2 Error)

## 🧠 Methodology

The neural network takes **(x, t)** as input and predicts **u(x, t)**.

The loss function combines:
- PDE residual loss
- Initial condition loss
- Boundary condition loss
- 
## 🏗️ Model Architecture

- Input: 2 neurons (x, t)
- Hidden layers: 8 layers (30 neurons each)
- Activation: Tanh
- Output: 1 neuron (u)

## 📊 Results

- ✔ Captures nonlinear behavior and shock formation
- ✔ Good agreement with analytical solution
- ✔ Relative L2 Error: ~0.02 – 0.1


## ⚙️ Installation

```bash
pip install -r requirements.txt
