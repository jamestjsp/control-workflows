# CLAUDE.md

## Project Overview

Control systems library using **ctrlsys** (C11 rewrite, formerly `slicot`, NOT slycot). Python 3.13+, managed with uv.

## CRITICAL: Use Library Functions

**NEVER reinvent control routines. ALWAYS use `control_workflows.lti`:**

```python
from control_workflows.lti import (
    # Models
    tf, ss, zpk, pid,
    TransferFunction, StateSpace, ZPK,
    # Conversions
    tf2ss, ss2tf, zpk2ss, ss2zpk, zpk2tf, tf2zpk,
    # Interconnections
    series, parallel, feedback,
    # Analysis
    is_controllable, is_observable,
    controllability_matrix, observability_matrix,
    ctrb_gramian, obsv_gramian, hankel_singular_values,
    # Design
    place, acker,
    # Frequency
    bode, nyquist, margin, freqresp,
    # Time response
    step_response, impulse_response, initial_response,
    # Reduction
    balreal, reduce, minreal,
    # Delays
    pade, absorbDelay, thiran, discretize_delay,
)
```

**Before implementing ANY control function, check if it exists in `lti/`.**

## Architecture

```
src/control_workflows/
├── lti/                      # REUSABLE LIBRARY - use these!
│   ├── transfer_function.py  # tf(), pid(), TransferFunction
│   ├── zpk.py                # zpk(), ZPK
│   ├── state_space.py        # ss(), StateSpace
│   ├── conversions.py        # tf2ss, ss2tf, zpk2ss, etc.
│   ├── interconnect.py       # series(), parallel(), feedback()
│   ├── analysis.py           # controllability, observability, gramians
│   ├── design.py             # place(), acker()
│   ├── frequency.py          # bode(), nyquist(), margin()
│   ├── response.py           # step_response(), impulse_response()
│   ├── reduction.py          # balreal(), reduce(), minreal()
│   └── delay.py              # pade(), absorbDelay(), thiran()
└── examples/                 # Demo code using the library
```

## Commands

```bash
# Run examples
uv run python -c "from control_workflows.examples.kalman_filtering import run_matlab_example; run_matlab_example()"
uv run python -c "from control_workflows.examples.dc_motor_control import run_matlab_example; run_matlab_example()"
uv run python -c "from control_workflows.examples.continuous_time_models import run_matlab_example; run_matlab_example()"
uv run python -c "from control_workflows.examples.control_system_modeling import run_matlab_example; run_matlab_example()"
uv run python -c "from control_workflows.examples.time_delay import run_matlab_example; run_matlab_example()"
uv run python -c "from control_workflows.examples.delay_to_z import run_matlab_example; run_matlab_example()"
```

## SLICOT Usage

**This project uses `ctrlsys` (C11), NOT `slycot` (Fortran).**

```bash
# Verify routine signature
uv run python -c "from ctrlsys import <routine>; help(<routine>)"
```

Arrays must be Fortran-ordered: `np.asfortranarray(A)`

Key routines:
- `sb02od` - Riccati solver
- `fb01vd` - Kalman filter recursion
- `ab04md` - Bilinear transformation
- `tb05ad` - Frequency response
- `tb04ad` - SS to TF
- `td04ad` - TF to SS
- `ab05md/nd/pd` - Series/feedback/parallel connections

## Skills

Use `/ctrlsys-control` for SLICOT routine lookup.
