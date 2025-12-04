# Constrained Policy Optimization (CPO)

Independent implementation of CPO for Isaac Lab.

## Structure
- `algorithm.py`: CPO algorithm implementation.
- `modules.py`: Actor-Critic network.
- `rewards.py`: Reward functions.
- `storage.py`: Rollout storage.
- `config/`: Environment and training configurations.

## Usage
Train:
```bash
python scripts/train.py --task Isaac-Velocity-Galileo-CPO-v0
```
*Check config/galileo_cpo_cfg.py for task ID.*

Play:
```bash
python scripts/play.py --task Isaac-Velocity-Galileo-CPO-v0 --model_path logs/.../model.pt
```
