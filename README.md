# computational-methods-in-optimization
Implementation of gradient-based optimization methods: least squares, projected GD, adaptive steps, and stochastic optimization.
# Least Squares Estimation & Gradient-Based Optimization Methods

This project explores numerical optimization techniques for solving polynomial
regression problems using both analytical and iterative methods. The work focuses
on solving a least-squares parameter estimation problem for noisy measurements,
followed by advanced gradient-based algorithms including projected gradient descent,
adaptive step-size methods, and stochastic gradient descent (SGD).

---

## Project Overview

Given noisy samples of a polynomial function  
f(x) = a₀ + a₁x + a₂x² + … + aₙxⁿ,  
the goal is to estimate the parameter vector **a** by minimizing a quadratic loss:

\[
h(a) = \frac{1}{2m} \| y - Xa \|_2^2
\]

Tasks include:
- Constructing the measurement matrix **X**
- Solving the least-squares problem analytically using the closed-form solution
- Implementing projected gradient descent on the constrained domain  
  \(\|a\|_2 \le r\)
- Comparing decaying step-size rules with AdaGrad
- Computing smoothness constants using eigenvalues of \(X^T X\)
- Implementing stochastic projected gradient descent (mini-batch)

---

## Methods Implemented

###  Analytical Least-Squares Solution
- Shows convexity of the objective function  
- Uses the closed-form estimator  
  \[
  a^* = (X^T X)^{-1} X^T y
  \]
- Demonstrates nearly exact reconstruction of the true coefficients  
  *(See Figure 4 in the report)* :contentReference[oaicite:2]{index=2}

---

###  Projected Gradient Descent (PGD)
- Projection onto ℓ₂-ball  
  \[
  \Pi_C(a) =
      \begin{cases}
      r \cdot \frac{a}{\|a\|_2}, & \|a\|_2 > r \\
      a, & \text{otherwise}
      \end{cases}
  \]
- Implemented with:
  - **Decaying step-size:** \(\eta_t = \frac{D}{G\sqrt{t}}\)
  - **AdaGrad adaptive step-size**

Results:
- Decaying step-size converges smoothly  
- AdaGrad may overshoot when early gradients are small (as seen in the error plots)  
  *(Figures 6–8)* :contentReference[oaicite:3]{index=3}

---

###  Smoothness Constant Computation
- The Lipschitz constant of ∇h(a) is:
  \[
  L = \lambda_{\max}\left(\frac{X^T X}{m}\right)
  \]
- Used to test step-sizes η = 1/(10L), 1/L, 10/L  
  *(Figure 8)* :contentReference[oaicite:4]{index=4}

---

###  Stochastic Gradient Descent (Mini-Batch)
- Implements SGD on the constrained domain  
- Evaluates multiple batch sizes: b = 1, 10, 100, 10,000
- Observations:
  - Smaller batch sizes → faster runtime but noisier gradient  
  - Larger batches → smoother convergence but slower  
  *(Final plots in the report)* :contentReference[oaicite:5]{index=5}

---

## Repository Contents

- **`optimization.ipynb`**  
  Full implementation of all analytical, gradient, and SGD algorithms.

- **`report.pdf`**  
  Contains mathematical derivations, explanations, and convergence plots.

---

## How to Run

1. Open `optimization.ipynb` in Jupyter Notebook or VS Code.
2. Install required packages:
   ```bash
   pip install numpy matplotlib
   ```
3. Run all code cells to reproduce the figures and compare optimization methods.

---

## Summary

This project gives hands-on experience with:
- Convex least-squares regression
- Projection methods in constrained optimization
- Adaptive step-size algorithms (AdaGrad)
- Smoothness/Lipschitz constants
- Stochastic optimization and runtime–accuracy trade-offs

These tools form the foundation for modern large-scale optimization and machine learning.


###  Open the Notebook

For a clean, interactive view of the optimization notebook, open it using nbviewer:

 **[Open `main.ipynb` in nbviewer](https://nbviewer.org/github/Carmel-Vald/computational-methods-in-optimization/blob/main/main.ipynb)**

