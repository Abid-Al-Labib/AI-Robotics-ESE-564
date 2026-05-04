from dataclasses import dataclass
from pathlib import Path


@dataclass
class LocalPickConfig:
    """Configuration for the local-pick RL task."""

    project_root: Path = Path(__file__).resolve().parents[3]
    xml_relative_path: str = "assets/mujoco/scene.xml"
    seed: int | None = None

    # Object/start-state curriculum. Start small, then widen this during experiments.
    fixed_object_position: tuple[float, float, float] | None = (0.5, -0.15, 0.125)
    object_xy_noise: float = 0.0
    approach_xy_noise: float = 0.0
    approach_z_noise: float = 0.0
    joint_noise: float = 0.0

    # Scene geometry.
    table_x_min: float = 0.43
    table_x_max: float = 0.57
    table_y_min: float = -0.20
    table_y_max: float = -0.10
    object_z: float = 0.125
    approach_height: float = 0.16
    side_approach_offset: float = 0.10
    side_grasp_height: float = 0.03
    lift_success_z: float = 0.145

    # Control.
    max_episode_steps: int = 75
    action_scale: float = 0.02
    frame_skip: int = 20
    open_gripper_ctrl: float = 255.0
    close_gripper_ctrl: float = 0.0
    side_grasp: bool = False

    # Reward shaping.
    reach_weight: float = 1.0
    lift_weight: float = 20.0
    success_bonus: float = 200.0
    action_penalty_weight: float = 0.01
    table_collision_penalty: float = 2.0
    align_close_bonus: float = 2.0
    align_close_threshold: float = 0.015
    premature_close_penalty: float = 5.0
    premature_close_threshold: float = 0.06
    contact_reward: float = 2.0
    hold_reward: float = 5.0
    hold_lift_threshold: float = 0.01

    @property
    def xml_path(self) -> Path:
        return self.project_root / self.xml_relative_path
