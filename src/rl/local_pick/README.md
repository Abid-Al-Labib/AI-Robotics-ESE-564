# Local Pick RL

This directory contains the isolated training setup for the local pick skill.
The training environment starts directly from a randomized approach pose above
the object, so it does not run perception, IK/RRT travel, or the full pipeline
on every episode.

## Training idea

Pipeline during training:

```text
reset directly to local pick approach
-> SAC policy outputs small joint-space deltas
-> scripted gripper closes after a fixed step
-> reward encourages reaching and lifting the object
```

Pipeline during runtime:

```text
perception + IK/RRT reaches pick approach
-> trained local pick model runs
-> object lifted check
-> continue to place phase
```

## First training run

From the repository root:

```bash
python src/rl/local_pick/train.py --timesteps 100000
```

Start with no randomization. Once that works, widen the curriculum:

```bash
python src/rl/local_pick/train.py --timesteps 300000 --object-noise 0.02 --approach-noise 0.01 --joint-noise 0.01
```

To train against the same broad table distribution used by `main.py`, use:

```bash
python src/rl/local_pick/train.py --timesteps 300000 --full-table-random --approach-noise 0.01 --joint-noise 0.01 --model-path models/local_pick_sac_full_table
```

## Visual evaluation

```bash
python src/rl/local_pick/evaluate.py --model-path models/local_pick_sac.zip
```

For full-table randomized evaluation:

```bash
python src/rl/local_pick/evaluate.py --model-path models/local_pick_sac_full_table.zip --full-table-random --episodes 10
```

## Align-only model

This version trains RL only to align the gripper near the object. Runtime then
uses the scripted gripper close and motor lift.

```bash
python src/rl/local_pick/train_align.py --timesteps 300000 --full-table-random --approach-noise 0.01 --joint-noise 0.01 --model-path models/local_pick_align_full_table
```

```bash
python src/rl/local_pick/evaluate_align.py --model-path models/local_pick_align_full_table.zip --full-table-random --episodes 10
```

## Notes

- The action is a 7D joint-space delta.
- The gripper is scripted at first to keep RL focused on local arm motion.
- The environment is headless during training and uses the viewer only for evaluation.
- The runtime observation layout must match the training observation layout exactly.

## Runtime integration sketch

After RRT moves to the pick approach pose, the trained policy can be used with
the live MuJoCo model/data:

```python
from pipeline.localPick.rl_local_pick import RLLocalPickController

local_pick = RLLocalPickController(env.model, env.data, "models/local_pick_sac.zip")
success = local_pick.execute(viewer)
```

The controller assumes the scene is already at the approach state and uses the
same observation/action layout as `LocalPickEnv`.
