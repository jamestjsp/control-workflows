# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Control systems library using **slicot** (C11 rewrite, NOT slycot) for numerical routines. Python 3.13+, managed with uv.

**IMPORTANT**: Import from `slicot`, not `slycot`:
```python
from slicot import sb02od  # correct
# from slycot import sb02od  # WRONG - different package
```

## Commands

```bash
# Kalman filtering example
uv run python -c "from control_workflows.examples.kalman_filtering import run_matlab_example; run_matlab_example()"

# DC motor control example
uv run python -c "from control_workflows.examples.dc_motor_control import run_matlab_example; run_matlab_example()"
```

## SLICOT Usage Patterns

Read routine docs via `uv run python -c "from slicot import <routine>; help(<routine>)"`.

SLICOT routines require Fortran-ordered arrays:
```python
A_f = np.asfortranarray(A)
```

Key routines:
- `sb02od` - Discrete/continuous Riccati solver (uses positional args: dico, jobb, fact, uplo, jobl, sort, n, m, p, A, B, Q, R, L, tol)
- `fb01vd` - Kalman filter recursion (returns upper triangular P - must symmetrize via `np.triu(P) + np.triu(P, 1).T`)
- `ab04md` - Bilinear transformation (continuous <-> discrete)
- `tf01md` - Discrete-time state-space simulation
- `tb05ad` - Frequency response / DC gain (use freq=0+0j for DC gain)

## Architecture

```
src/control_workflows/
├── examples/
│   ├── kalman_filtering/
│   │   ├── steady_state.py   # sb02od for filter ARE
│   │   ├── time_varying.py   # fb01vd for recursive updates
│   │   └── matlab_example.py # Example runner
│   └── dc_motor_control/
│       ├── model.py          # DC motor state-space model
│       ├── lqr.py            # sb02od for LQR gain
│       └── matlab_example.py # Feedforward vs integral vs LQR comparison
```

## Skills

Use the `slicot-control` skill for SLICOT routine lookup and usage guidance:
```
/slicot-control
```

Install via: `claude plugin install slicot-control`
