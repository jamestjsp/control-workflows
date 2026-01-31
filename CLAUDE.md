# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Control systems library using **slicot** (C11 rewrite, NOT slycot) for numerical routines. Python 3.13+, managed with uv.

**CRITICAL**: This project uses `slicot` (C11 rewrite), NOT `slycot` (Fortran wrapper). They have **different APIs**:
```python
from slicot import sb02od  # correct
# from slycot import sb02od  # WRONG - different package, different signature
```

## Commands

```bash
# Kalman filtering example
uv run python -c "from control_workflows.examples.kalman_filtering import run_matlab_example; run_matlab_example()"

# DC motor control example
uv run python -c "from control_workflows.examples.dc_motor_control import run_matlab_example; run_matlab_example()"

# Continuous-time model creation example
uv run python -c "from control_workflows.examples.continuous_time_models import run_matlab_example; run_matlab_example()"

# Control system modeling (block diagram algebra)
uv run python -c "from control_workflows.examples.control_system_modeling import run_matlab_example; run_matlab_example()"
```

## SLICOT Usage Patterns

**ALWAYS verify routine signature before use** - slicot API differs from slycot:
```bash
uv run python -c "from slicot import <routine>; help(<routine>)"
```

SLICOT routines require Fortran-ordered arrays:
```python
A_f = np.asfortranarray(A)
```

Key routines (signatures verified for slicot, NOT slycot):
- `sb02od` - Discrete/continuous Riccati solver (dico, jobb, fact, uplo, jobl, sort, n, m, p, A, B, Q, R, L, tol)
- `fb01vd` - Kalman filter recursion (returns upper triangular P - must symmetrize via `np.triu(P) + np.triu(P, 1).T`)
- `ab04md` - Bilinear transformation (continuous <-> discrete)
- `tf01md` - Discrete-time state-space simulation
- `tb05ad` - Frequency response / DC gain (use freq=0+0j for DC gain)
- `tb04ad` - State-space to transfer function (rowcol, A, B, C, D)
- `td04ad` - Transfer function to state-space minimal realization (rowcol, m, p, index, dcoeff, ucoeff, tol)
- `ab05md` - Cascade/series connection (uplo, over, A1, B1, C1, D1, A2, B2, C2, D2)
- `ab05nd` - Feedback connection (over, alpha, A1, B1, C1, D1, A2, B2, C2, D2) - NOTE: use alpha=+1 for negative feedback
- `ab05pd` - Parallel connection (n1, m, p, n2, alpha, A1, B1, C1, D1, A2, B2, C2, D2)

## Architecture

```
src/control_workflows/
├── lti/                      # LTI model representations
│   ├── transfer_function.py  # tf(), pid(), TransferFunction, LaplaceDomain
│   ├── zpk.py                # zpk(), ZPK (zero-pole-gain)
│   ├── state_space.py        # ss(), StateSpace
│   └── conversions.py        # tf2ss, ss2tf, zpk2tf, etc (uses tb04ad, td04ad)
├── examples/
│   ├── kalman_filtering/
│   │   ├── steady_state.py   # sb02od for filter ARE
│   │   ├── time_varying.py   # fb01vd for recursive updates
│   │   └── matlab_example.py # Example runner
│   ├── dc_motor_control/
│   │   ├── model.py          # DC motor state-space model
│   │   ├── lqr.py            # sb02od for LQR gain
│   │   └── matlab_example.py # Feedforward vs integral vs LQR comparison
│   ├── continuous_time_models/
│   │   └── matlab_example.py # TF, ZPK, SS creation and conversion demo
│   └── control_system_modeling/
│       ├── interconnect.py   # series(), parallel(), feedback() using ab05xx
│       └── matlab_example.py # Mixed model types, block diagram algebra
```

## Skills

Use the `slicot-control` skill for SLICOT routine lookup and usage guidance:
```
/slicot-control
```

Install via: `claude plugin install slicot-control`
