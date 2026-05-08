# Option A — Progress Reward + Reached-Box Flag

Inspired by DeepMind MuJoCo Playground `PandaPickCubeCartesian`.
Drop-in reward redesign for `LocalPickCartesianDirectLiftEnv`.
No architecture change — same SAC, same SB3, same camera-based obs.

---

## Core Ideas

### 1. Progress Reward
Only reward *improvement* over the best reward seen so far in the episode.
Kills every farming exploit by construction — you cannot gain reward by
staying in the same state twice.

```python
total_reward = compute_raw_reward(...)
reward = max(total_reward - prev_best_reward, 0)
prev_best_reward = max(total_reward, prev_best_reward)
```

### 2. reached_box_flag
A latch that flips True the first time the gripper gets within threshold
distance of the bottle. Before the flag: only approach reward is active.
After the flag: lift reward activates. Clean two-phase design — no height
gating, no gripper_opening threshold, no camera-stale issues.

```python
if not reached_box and gt_distance < 0.012:
    reached_box = True  # latches, never resets within episode

if not reached_box:
    reward = 4.0 * (1 - tanh(5 * gt_distance))   # phase 1: approach
else:
    reward = 8.0 * (1 - tanh(5 * height_error))   # phase 2: lift
    reward += lifted_bonus if object_lift > 0.02
    reward += success_bonus if success
```

---

## What Changes

- New env: `LocalPickCartesianDirectLiftEnv` reward replaced with the above
- `_prev_best_reward` tracked in episode state (reset each episode)
- `_reached_box` flag tracked in episode state (reset each episode)
- GT distance used for flag and phase-2 reward (camera used for obs only)

## What Stays the Same

- SAC + SB3 + VecNormalize
- Camera-based observations (perception-based — satisfies rubric)
- Cartesian action space
- 24 parallel envs, 1M timesteps overnight

## Training Command

```
python src\rl\local_pick\train_cartesian_direct_lift.py \
  --timesteps 1000000 --n-envs 24 \
  --model-path models/local_pick_option_a_1m \
  --object-noise 0.02 --approach-noise 0.01 \
  --approach-z-noise 0.01 --joint-noise 0.01
```

## Risk

Medium. Progress reward is proven (DeepMind uses it). Two-phase reward is
clean. But 1M steps at 24 envs may not be enough sample budget. Overnight
run — eval in morning.
