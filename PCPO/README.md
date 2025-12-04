# Projection-Based CPO (PCPO)

Implementation of Projection-Based Constrained Policy Optimization.
Inherits core logic from CPO.

## Structure
- `algorithm.py`: PCPO implementation (extends CPO).
- `modules.py`: Actor-Critic network.
- `rewards.py`: Reward functions.
- `config/`: Environment and training configurations.

## Usage
Train:
```bash
python scripts/train.py --task Isaac-Velocity-Galileo-PCPO-v0
```
*Check config/galileo_pcpo_cfg.py for task ID.*
