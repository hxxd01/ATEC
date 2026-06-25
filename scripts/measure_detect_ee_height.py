#!/usr/bin/env python3
"""Place trash under gripper_base (detect hold pose), measure distance, save camera views."""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_DEMO = os.path.join(_ROOT, "demo")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--place-settle-steps", type=int, default=30)
parser.add_argument(
    "--save-dir",
    type=str,
    default="logs/detect_hold_under_ee",
    help="Directory for PNG snapshots (head / ee / ground / panel).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.save_dir:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_utils import draw_text_overlay, ground_camera_follow, obs_image_rgb, to_uint8_hwc  # noqa: E402
from solution import AlgSolution  # noqa: E402

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    imageio = None

NUM_OBJECTS = 18
GRASP_DIST = 0.20
FAR_AWAY = (-120.0, -120.0, -5.0)

TRASH_TYPES = (
    ("sugar", "object_1", 0.15, (0.0, 0.707, 0.0, 0.707)),
    ("mustard", "object_7", 0.10, (0.0, 0.0, -0.707, 0.707)),
    ("banana", "object_13", 0.10, (0.0, 0.0, -0.707, 0.707)),
)


def _enable_ground_view_camera(env_cfg) -> None:
    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils

    if not hasattr(env_cfg, "scene"):
        return
    env_cfg.scene.ground_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ground_camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    img = getattr(getattr(env_cfg, "observations", None), "image", None)
    if img is not None:
        img.ground_rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("ground_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )


def _detect_hold_arm(agent: AlgSolution) -> list[float]:
    return [float(x) for x in agent._detect_hold_arm_cmd()]


def _ee_pos(robot, ee_id: int) -> np.ndarray:
    return robot.data.body_pos_w[0, ee_id, :3].detach().cpu().numpy()


def _root_z(robot) -> float:
    return float(robot.data.root_pos_w[0, 2].detach().cpu())


def _make_action(agent: AlgSolution, obs, leg_env_12: list[float], arm8: list[float]) -> torch.Tensor:
    proprio = obs["proprio"].to(agent.device)
    adim = (int(proprio.shape[-1]) - 12) // 3
    action = torch.zeros((1, adim), device=agent.device, dtype=torch.float32)
    action[0, :12] = torch.tensor(leg_env_12, device=agent.device, dtype=torch.float32)
    action[0, 12:20] = torch.tensor(arm8, device=agent.device, dtype=torch.float32)
    return action


def _policy_leg_action(
    agent: AlgSolution,
    obs,
    *,
    use_down_policy: bool,
    cmd: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[float]:
    proprio = obs["proprio"].to(agent.device)
    adim = (int(proprio.shape[-1]) - 12) // 3
    policy = agent._lock_grasp_policy if use_down_policy and agent._lock_grasp_policy is not None else agent.policy
    policy_obs = agent._extract_policy_obs(obs, adim, cmd_override=cmd)
    with torch.inference_mode():
        leg_action = policy(policy_obs)
    leg_action = torch.as_tensor(leg_action, device=agent.device, dtype=torch.float32)
    if leg_action.ndim == 1:
        leg_action = leg_action.unsqueeze(0)
    leg_env = agent._map_policy_action_to_env_action(leg_action, adim)
    return leg_env[0, :12].detach().cpu().tolist()


def _settle_pose(env, agent: AlgSolution, obs, *, use_down_policy: bool, steps: int) -> dict:
    arm = _detect_hold_arm(agent)
    for _ in range(steps):
        legs = _policy_leg_action(agent, obs, use_down_policy=use_down_policy)
        obs, _, _, _, _ = env.step(_make_action(agent, obs, legs, arm))
    return obs


def _write_object_pose(env, obj_name: str, pos_xyz: tuple[float, float, float], quat_wxyz: tuple[float, ...]) -> None:
    obj = env.unwrapped.scene[obj_name]
    dev = obj.device
    pos = torch.tensor([[pos_xyz[0], pos_xyz[1], pos_xyz[2]]], device=dev, dtype=torch.float32)
    quat = torch.tensor([list(quat_wxyz)], device=dev, dtype=torch.float32)
    obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
    obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev, dtype=torch.float32))


def _hide_other_objects(env, keep_name: str) -> None:
    for i in range(1, NUM_OBJECTS + 1):
        name = f"object_{i}"
        if name == keep_name:
            continue
        _write_object_pose(env, name, FAR_AWAY, (1.0, 0.0, 0.0, 0.0))


def _label_panel(img: np.ndarray, label: str) -> np.ndarray:
    try:
        import cv2

        out = np.ascontiguousarray(img.copy())
        cv2.putText(out, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
        return out
    except Exception:
        return img


def _save_png(path: str, frame: np.ndarray | None) -> bool:
    if frame is None or imageio is None:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, frame)
    return True


def _save_views(
    env,
    agent: AlgSolution,
    obs,
    *,
    use_down_policy: bool,
    save_dir: str,
    stem: str,
    overlay_lines: list[str],
) -> list[str]:
    if not save_dir or imageio is None:
        return []

    ground_camera_follow(env)
    legs_hold = _policy_leg_action(agent, obs, use_down_policy=use_down_policy)
    arm = _detect_hold_arm(agent)
    obs, _, _, _, _ = env.step(_make_action(agent, obs, legs_hold, arm))

    saved: list[str] = []
    views = {
        "head": obs_image_rgb(obs, "head_rgb"),
        "ee": obs_image_rgb(obs, "ee_rgb"),
        "ground": obs_image_rgb(obs, "ground_rgb"),
    }
    for key, frame in views.items():
        if frame is None:
            continue
        frame = draw_text_overlay(frame, overlay_lines)
        path = os.path.join(save_dir, f"{stem}_{key}.png")
        if _save_png(path, frame):
            saved.append(path)

    panels = [f for f in (views["ground"], views["head"], views["ee"]) if f is not None]
    if panels:
        h = max(p.shape[0] for p in panels)
        resized = []
        labels = ("ground", "head", "ee")
        for p, lab in zip(panels, labels):
            if p.shape[0] != h:
                scale = h / p.shape[0]
                w = max(1, int(p.shape[1] * scale))
                try:
                    import cv2

                    p = cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass
            p = draw_text_overlay(p, overlay_lines)
            resized.append(_label_panel(p, lab))
        panel = np.concatenate(resized, axis=1)
        panel_path = os.path.join(save_dir, f"{stem}_panel.png")
        if _save_png(panel_path, panel):
            saved.append(panel_path)
    return saved


def _measure_under_ee(
    env,
    agent: AlgSolution,
    robot,
    ee_id: int,
    *,
    posture: str,
    use_down_policy: bool,
    trash_name: str,
    obj_name: str,
    spawn_z: float,
    quat_wxyz: tuple[float, ...],
    settle_steps: int,
    place_settle_steps: int,
    save_dir: str,
) -> None:
    obs, _ = env.reset()
    agent.reset()

    obs = _settle_pose(env, agent, obs, use_down_policy=use_down_policy, steps=settle_steps)
    ee_before = _ee_pos(robot, ee_id)
    root_z = _root_z(robot)

    _hide_other_objects(env, obj_name)
    _write_object_pose(
        env,
        obj_name,
        (float(ee_before[0]), float(ee_before[1]), spawn_z),
        quat_wxyz,
    )

    arm = _detect_hold_arm(agent)
    legs = _policy_leg_action(agent, obs, use_down_policy=use_down_policy)
    for _ in range(place_settle_steps):
        obs, _, _, _, _ = env.step(_make_action(agent, obs, legs, arm))
        legs = _policy_leg_action(agent, obs, use_down_policy=use_down_policy)

    ee = _ee_pos(robot, ee_id)
    obj = env.unwrapped.scene[obj_name]
    obj_pos = obj.data.root_pos_w[0, :3].detach().cpu().numpy()
    dist3d = float(np.linalg.norm(ee - obj_pos))
    dist_xy = float(np.hypot(ee[0] - obj_pos[0], ee[1] - obj_pos[1]))
    dz = float(ee[2] - obj_pos[2])
    score_ok = dist3d <= GRASP_DIST

    overlay = [
        f"{posture}/{trash_name}",
        f"dist3d={dist3d:.3f} score={'YES' if score_ok else 'NO'}",
        f"ee_z={ee[2]:.3f} obj_z={obj_pos[2]:.3f} dz={dz:+.3f}",
        f"dist_xy={dist_xy:.3f}",
    ]
    print(
        f"[{posture}/{trash_name}] root_z={root_z:.3f} "
        f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
        f"obj=({obj_pos[0]:.3f},{obj_pos[1]:.3f},{obj_pos[2]:.3f}) "
        f"dist_xy={dist_xy:.4f} dz={dz:+.3f} dist3d={dist3d:.3f} score_ok={score_ok}",
        flush=True,
    )

    if save_dir:
        stem = f"{posture}_{trash_name}"
        saved = _save_views(
            env,
            agent,
            obs,
            use_down_policy=use_down_policy,
            save_dir=save_dir,
            stem=stem,
            overlay_lines=overlay,
        )
        for p in saved:
            print(f"[save] {p}", flush=True)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.observations.proprio.enable_corruption = False
    env_cfg.observations.extero.enable_corruption = False
    if args_cli.save_dir:
        _enable_ground_view_camera(env_cfg)
    else:
        if hasattr(env_cfg, "scene"):
            env_cfg.scene.head_camera = None
            env_cfg.scene.ee_camera = None
        if hasattr(env_cfg, "observations"):
            env_cfg.observations.image = None

    save_dir = os.path.abspath(args_cli.save_dir) if args_cli.save_dir else ""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if imageio is None:
            raise ImportError("imageio required for --save-dir. pip install imageio")

    env = gym.make(args_cli.task, cfg=env_cfg)
    agent = AlgSolution()
    agent.bind_env(env)

    robot = env.unwrapped.scene.articulations["robot"]
    ee_ids, ee_names = robot.find_bodies("gripper_base")
    ee_id = int(ee_ids[0])
    print(
        f"[info] ee={ee_names[0]} detect_arm={_detect_hold_arm(agent)[:3]} "
        f"save_dir={save_dir or 'none'}",
        flush=True,
    )

    for posture, use_down in (("squat-down", True), ("stand", False)):
        print(f"\n=== posture={posture} ===", flush=True)
        for trash_name, obj_name, spawn_z, quat in TRASH_TYPES:
            _measure_under_ee(
                env,
                agent,
                robot,
                ee_id,
                posture=posture,
                use_down_policy=use_down,
                trash_name=trash_name,
                obj_name=obj_name,
                spawn_z=spawn_z,
                quat_wxyz=quat,
                settle_steps=args_cli.settle_steps,
                place_settle_steps=args_cli.place_settle_steps,
                save_dir=save_dir,
            )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
