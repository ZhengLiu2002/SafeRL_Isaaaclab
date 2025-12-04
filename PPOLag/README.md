# PPO-Lagrange (PPOLag)

Independent implementation of PPO-Lagrange for Isaac Lab.

## Structure
- `algorithm.py`: PPOLag algorithm implementation.
- `modules.py`: Actor-Critic network.
- `rewards.py`: Reward functions.
- `storage.py`: Rollout storage.
- `config/`: Environment and training configurations.

## Usage
Train:
```bash
python scripts/train.py --task Isaac-Velocity-Galileo-PPOLag-v0
```

Play:
```bash
python scripts/play.py --task Isaac-Velocity-Galileo-PPOLag-v0 --model_path logs/.../model.pt
```
