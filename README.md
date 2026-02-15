# control-workflows

Agent skills for control system design and analysis workflows, built on [ctrlsys](https://github.com/jamestjsp/ctrlsys).

> **Note**: This project is in early development. More workflows will be added over time.

## Getting Started

```bash
git clone https://github.com/jamestjsp/control-workflows.git
cd control-workflows
uv sync
```

Requires Python 3.13+.

## Usage with AI Agents

This library is designed for use with AI coding agents (Claude Code, Cursor, GitHub Copilot). The recommended approach:

### 1. Install Skills

```bash
claude plugin install ctrlsys-control
claude plugin install control-theory
```

### 2. Use Skills in Your Session

```
/ctrlsys-control
```

Skills provide routine lookup, usage patterns, and control theory guidance.

## Reading Routine Documentation

Every ctrlsys routine has detailed docstrings. Access them via:

```bash
uv run python -c "from ctrlsys import <routine>; help(<routine>)"
```

Example:
```bash
uv run python -c "from ctrlsys import sb02od; help(sb02od)"
```

This displays parameters, return values, mathematical formulations, and usage examples.

## Important: ctrlsys vs slycot

This project uses **ctrlsys** (C11 implementation), not slycot:

```python
from ctrlsys import sb02od  # correct
# from slycot import sb02od  # wrong package
```

## License

MIT
