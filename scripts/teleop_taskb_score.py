#!/usr/bin/env python3
"""Task B keyboard teleop with final platform-touch score printout."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from isaaclab.app import AppLauncher


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


parser = argparse.ArgumentParser(description="Task B teleop (keyboard) + final score.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--ll_policy", type=str, default="demo/down_policy.pt")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--grasp_dist", type=float, default=0.20)
parser.add_argument("--vx_max", type=float, default=2.0)
parser.add_argument("--vy_max", type=float, default=1.0)
parser.add_argument("--wz_max", type=float, default=1.0)
parser.add_argument("--print_every", type=int, default=20, help="Print score every N steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Keyboard teleop requires viewport focus.
if getattr(args_cli, "headless", False):
    raise ValueError("Keyboard teleop needs viewer. Remove --headless.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import omni.appwindow as appwindow  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401,E402
from atec_rl_lab.train.locomotion.velocity.mdp.events import DETECT_HOLD_ARM_ACTION  # noqa: E402
from demo.teleop_controller import TaskDTeleopController  # noqa: E402


class TaskBTouchScoreTracker:
    """Count one-time touch score: EE-object 3D dist <= threshold."""

    def __init__(self, env, threshold: float = 0.20, ee_body_name: str = "gripper_base", num_objects: int = 18):
        self.env = env
        self.threshold = float(threshold)
        self.ee_body_name = ee_body_name
        self.num_objects = int(num_objects)
        self._ee_body_idx = None
        self.scored = None
        self.total_score = 0

    def reset(self) -> None:
        self.scored = torch.zeros((self.env.num_envs, self.num_objects), device=self.env.device, dtype=torch.bool)
        self.total_score = 0

    def _ee_pos_w(self) -> torch.Tensor:
        robot = self.env.scene["robot"]
        if self._ee_body_idx is None:
            body_ids, names = robot.find_bodies(self.ee_body_name)
            if len(body_ids) == 0:
                raise ValueError(f"Cannot find ee body '{self.ee_body_name}'")
            self._ee_body_idx = int(body_ids[0])
            print(f"[teleop] ee body = {names[0]} (idx={self._ee_body_idx})", flush=True)
        return robot.data.body_pos_w[:, self._ee_body_idx, :3]

    def _obj_pos_w(self) -> torch.Tensor:
        objs = []
        for i in range(1, self.num_objects + 1):
            obj = self.env.scene[f"object_{i}"]
            objs.append(obj.data.root_pos_w[:, :3])
        return torch.stack(objs, dim=1)

    def update(self) -> int:
        ee = self._ee_pos_w()
        obj = self._obj_pos_w()
        dist = torch.linalg.norm(obj - ee.unsqueeze(1), dim=-1)
        reached = dist <= self.threshold
        newly = reached & (~self.scored)
        self.scored |= reached
        inc = int(newly.sum().item())
        if inc > 0:
            self.total_score += inc
        return inc

    def scored_count_env0(self) -> int:
        return int(self.scored[0].sum().item())


def _resolve_policy(path: str) -> str:
    raw = os.path.expanduser(path)
    cands = [raw]
    if not os.path.isabs(raw):
        cands.append(os.path.join(_ROOT, raw))
    for c in cands:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"ll_policy not found: {path}")


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if args_cli.num_envs != 1:
        print("[teleop] forcing score print for env0 only.", flush=True)

    policy_path = _resolve_policy(args_cli.ll_policy)
    controller = TaskDTeleopController(
        policy_path=policy_path,
        device=args_cli.device,
        vx_min=-abs(float(args_cli.vx_max)),
        vx_max=abs(float(args_cli.vx_max)),
        vy_max=abs(float(args_cli.vy_max)),
        wz_max=abs(float(args_cli.wz_max)),
    )
    # Keep arm in Task-B detect pose (same as training wrapper).
    controller.arm_default_action = torch.tensor(
        DETECT_HOLD_ARM_ACTION, device=controller.device, dtype=torch.float32
    ).view(1, -1)

    tracker = TaskBTouchScoreTracker(env.unwrapped, threshold=float(args_cli.grasp_dist))
    obs, _ = env.reset()
    tracker.reset()
    controller.reset()

    input_iface = carb.input.acquire_input_interface()
    kb = appwindow.get_default_app_window().get_keyboard()
    keys = {
        carb.input.KeyboardInput.W: False,
        carb.input.KeyboardInput.S: False,
        carb.input.KeyboardInput.A: False,
        carb.input.KeyboardInput.D: False,
        carb.input.KeyboardInput.Q: False,
        carb.input.KeyboardInput.E: False,
    }

    def _on_key(event, *_args):
        if event.input in keys:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                keys[event.input] = True
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                keys[event.input] = False
        return True

    sub = input_iface.subscribe_to_keyboard_events(kb, _on_key)
    print("[teleop] controls: W/S vx, A/D vy, Q/E yaw, ESC close app.", flush=True)

    step = 0
    try:
        while simulation_app.is_running():
            step += 1
            vx = float(keys[carb.input.KeyboardInput.W]) - float(keys[carb.input.KeyboardInput.S])
            vy = float(keys[carb.input.KeyboardInput.D]) - float(keys[carb.input.KeyboardInput.A])
            wz = float(keys[carb.input.KeyboardInput.Q]) - float(keys[carb.input.KeyboardInput.E])
            controller.set_velocity_command(vx * args_cli.vx_max, vy * args_cli.vy_max, wz * args_cli.wz_max)

            with torch.inference_mode():
                resp = controller.predicts(obs, 0.0)
                actions = torch.tensor(resp["action"], dtype=torch.float32, device=args_cli.device).view(1, -1)
                obs, _rew, term, trunc, _info = env.step(actions)

            inc = tracker.update()
            if inc > 0:
                print(
                    f"[teleop] +{inc} touch(es), total_score={tracker.total_score}, env0_scored={tracker.scored_count_env0()}/18",
                    flush=True,
                )
            if step % int(args_cli.print_every) == 0:
                print(f"[teleop] step={step} env0_scored={tracker.scored_count_env0()}/18", flush=True)

            done = bool(term.item() if hasattr(term, "item") else term) or bool(
                trunc.item() if hasattr(trunc, "item") else trunc
            )
            if done:
                print(
                    f"[teleop] episode done. FINAL SCORE: {tracker.total_score} (env0_scored={tracker.scored_count_env0()}/18)",
                    flush=True,
                )
                break
            # small sleep to keep key handling responsive
            time.sleep(0.001)
    finally:
        input_iface.unsubscribe_from_keyboard_events(kb, sub)
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
