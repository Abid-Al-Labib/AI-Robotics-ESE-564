# RL Pick Pipeline — Review Notes

## Critical (causing transfer/drop failures)

### 1. IK solution selection mismatch
- **Training** (`_sample_approach_joint_state`): picks IK solution closest to `home = [0, 0, 0, -1.57, 0, 1.57, -0.785]`
- **Pipeline** (`get_ik_goal`): picks IK solution closest to `q_current` (wherever RRT left the arm)
- **Effect**: joint7 (wrist) arrives at a completely different angle → gripper is rotated → policy trained on one orientation, deployed at another. Biggest cause of drops.
- **Fix**: In `get_ik_goal`, bias IK selection toward home config for pick approach.

### 2. Approach height mismatch
- **Training**: starts arm at `object_z + approach_height = 0.125 + 0.16 = 0.285m`
- **Pipeline**: uses `perceived_z + 0.15`, and perception reads bottle top (~0.151m) → approach ~0.301m
- **Effect**: arm starts ~1.6cm higher than training distribution from step 1.
- **Fix**: Change pipeline pick `z_offset` to `0.16`, or change `approach_height` to `0.15` in config.

### 3. Perception noise larger than training noise
- **Training**: `--approach-noise 0.008` (8mm XY)
- **Pipeline**: camera gives ~2–3cm XY error (visible in logs)
- **Effect**: `ee_pos - object_pos` vector is noisier than anything policy trained on.
- **Fix**: Increase `--approach-noise` to `0.025` when retraining.

---

## Minor Bugs

### 4. `fixed_object_position` z=0.13 vs `object_z=0.125` everywhere else
- `config.py` default fixed position: `(0.4, -0.15, 0.13)` → z=0.13
- Scene XML, `mujoco_env.py`, random sampling all use z=0.125
- **Fix**: Change `fixed_object_position` default to `(0.4, -0.15, 0.125)`.

### 5. Training table x-bounds narrower than pipeline
- Training: x ∈ [0.38, 0.62]
- Pipeline (`mujoco_env.py`): x ∈ [0.35, 0.65]
- **Fix**: Align training bounds to match pipeline.

### 6. `_table_collision_penalty` uses hardcoded body IDs
- `robot_body_ids = set(range(1, 12))` — silently breaks if XML body order changes.
- **Fix**: Look up robot body IDs by name.

---

## Training Improvements

### 7. `gamma=0.98` too low for 120-step episodes
- Effective horizon: 1/(1-0.98) = 50 steps. Rewards after step 50 heavily discounted.
- Policy undervalues late-episode rewards (success bonus, hold bonus at full lift).
- **Fix**: Change to `gamma=0.99`.

### 8. `gradient_steps=1`
- SAC can use `gradient_steps=2` with no extra env interaction.
- Better critic stability for precise manipulation.
- **Fix**: Set `gradient_steps=2` in `train.py`.

---

## Recommended Retrain Command (after applying fixes)
```bash
python src/rl/local_pick/train.py \
  --timesteps 600000 \
  --model-path models/local_pick_sac_v2 \
  --full-table-random \
  --object-noise 0.01 \
  --approach-noise 0.025 \
  --joint-noise 0.01
```
