# Phase E: ODE Integration Accuracy

Comparison of Euler, RK4, and DOPRI5 on the harmonic oscillator:
  dx/dt = v, dv/dt = -x
  x(0) = 1, v(0) = 0
  Analytical: x(t) = cos(t), v(t) = -sin(t)

Integration interval: [0, 10]. True x(10) = -0.83907153

## Results Table

| Method | dt | Steps | x(10) | v(10) | MSE |
|--------|-----|-------|-------|-------|-----|
| euler | 0.100 | 100 | -1.40884698 | 0.84850693 | 2.09e-01 |
| euler | 0.050 | 200 | -1.08282636 | 0.68933296 | 4.03e-02 |
| euler | 0.010 | 1000 | -0.88228002 | 0.57161820 | 1.31e-03 |
| rk4 | 0.100 | 100 | -0.83907546 | 0.54401377 | 3.47e-11 |
| rk4 | 0.050 | 200 | -0.83907179 | 0.54402066 | 1.36e-13 |
| rk4 | 0.010 | 1000 | -0.83907153 | 0.54402111 | 3.47e-19 |
| dopri5 | 0.100 | 100 | -0.83907150 | 0.54402110 | 3.88e-16 |
| dopri5 | 0.050 | 200 | -0.83907153 | 0.54402111 | 3.77e-19 |
| dopri5 | 0.010 | 1000 | -0.83907153 | 0.54402111 | 3.83e-26 |

## Analysis

- **Euler** (1st order): Error ~O(dt). Large errors even at dt=0.01.
  The harmonic oscillator is a tough test because Euler introduces
  artificial energy gain, causing the solution to spiral outward.

- **RK4** (4th order): Error ~O(dt^4). Much more accurate than Euler.
  At dt=0.01, error is negligible for practical purposes.

- **DOPRI5** (5th order): Error ~O(dt^5). Most accurate at each dt.
  The Dormand-Prince method achieves roughly 10x lower error than RK4
  at the same step size. In adaptive mode, it would also control the
  step size automatically to maintain a specified tolerance.

## Reproduction

```bash
python -c "from hhcra.verification import run_phase_e; run_phase_e()"
```
