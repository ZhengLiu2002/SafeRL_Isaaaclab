# First-Order CPO (FOCPO)

Implementation of First-Order Constrained Policy Optimization.
Inherits core logic from CPO but uses first-order approximations.

## Structure
- `algorithm.py`: FOCPO implementation (extends CPO).
- `modules.py`: Actor-Critic network.
- `rewards.py`: Reward functions.
- `config/`: Environment and training configurations.

## Usage
Train:
```bash
python scripts/train.py --task Isaac-Velocity-Galileo-FOCPO-v0
```
*Check config/galileo_focpo_cfg.py for task ID.*
