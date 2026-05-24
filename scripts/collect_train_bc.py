import argparse
from collections import deque
import importlib
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from isaaclab.app import AppLauncher


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 4, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).flatten(1))


class TaskDBCGRUPolicy(nn.Module):
    def __init__(self, proprio_dim: int = 9, cmd_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_encoder = ConvEncoder(in_ch=4, out_dim=128)
        self.ee_encoder = ConvEncoder(in_ch=4, out_dim=128)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(nn.Linear(128 + 128 + 64, hidden_dim), nn.ReLU(inplace=True))
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, cmd_dim))

    def forward_step(
        self,
        head_img: torch.Tensor,  # [B,4,H,W]
        ee_img: torch.Tensor,    # [B,4,H,W]
        proprio: torch.Tensor,   # [B,9]
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_feat = self.head_encoder(head_img)
        e_feat = self.ee_encoder(ee_img)
        p_feat = self.proprio_mlp(proprio)
        fused = self.fuse(torch.cat([h_feat, e_feat, p_feat], dim=-1)).unsqueeze(1)
        out, hidden = self.gru(fused, hidden)
        return self.head(out[:, -1, :]), hidden


class ScoredEpisodeReplayBuffer:
    """Global replay of full episodes with episode_score > min_episode_score."""

    def __init__(self, num_envs: int, replay_capacity: int, min_episode_score: float = 0.0):
        self.num_envs = int(num_envs)
        self.replay_capacity = int(replay_capacity)
        self.min_episode_score = float(min_episode_score)
        self.scored: deque[dict] = deque(maxlen=self.replay_capacity)
        self.active = [
            {"head": [], "ee": [], "prop": [], "cmd": [], "done": []}
            for _ in range(self.num_envs)
        ]
        self.total_finished = 0
        self.total_scored_added = 0
        self.total_discarded = 0

    def add_batch(self, head_u8, ee_u8, prop, cmd, done):
        for env_id in range(self.num_envs):
            a = self.active[env_id]
            a["head"].append(head_u8[env_id].copy())
            a["ee"].append(ee_u8[env_id].copy())
            a["prop"].append(prop[env_id].copy())
            a["cmd"].append(cmd[env_id].copy())
            a["done"].append(bool(done[env_id]))

    def _build_episode(self, env_id: int) -> dict | None:
        a = self.active[env_id]
        if len(a["head"]) == 0:
            return None
        return {
            "head": np.stack(a["head"], axis=0).astype(np.uint8),
            "ee": np.stack(a["ee"], axis=0).astype(np.uint8),
            "prop": np.stack(a["prop"], axis=0).astype(np.float32),
            "cmd": np.stack(a["cmd"], axis=0).astype(np.float32),
            "done": np.asarray(a["done"], dtype=np.bool_),
        }

    def _reset_active(self, env_id: int):
        self.active[env_id] = {"head": [], "ee": [], "prop": [], "cmd": [], "done": []}

    def finalize_done(
        self,
        newly_done_mask: np.ndarray,
        scores: np.ndarray,
        min_len: int = 1,
    ):
        for env_id in np.nonzero(newly_done_mask)[0].tolist():
            env_id = int(env_id)
            score = float(scores[env_id])
            ep = self._build_episode(env_id)
            self._reset_active(env_id)
            self.total_finished += 1
            if ep is None:
                self.total_discarded += 1
                continue
            ep_len = int(ep["prop"].shape[0])
            if ep_len < int(min_len) or score <= self.min_episode_score:
                self.total_discarded += 1
                continue
            ep["score"] = score
            ep["env_id"] = env_id
            ep["length"] = ep_len
            self.scored.append(ep)
            self.total_scored_added += 1

    def __len__(self) -> int:
        return len(self.scored)

    def sample_episodes(self, batch_size: int, min_len: int = 1):
        avail = [ep for ep in self.scored if int(ep["prop"].shape[0]) >= int(min_len)]
        if len(avail) == 0:
            return []
        pick = min(int(batch_size), len(avail))
        idx = np.random.choice(len(avail), size=pick, replace=False)
        return [avail[int(i)] for i in idx]


def _to_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device)
    return torch.as_tensor(x, device=device)


def _prep_rgb(x, image_size: int, device):
    t = _to_torch(x, device).float()
    if t.ndim != 4:
        raise ValueError(f"Unexpected RGB dims: {tuple(t.shape)}")
    if t.shape[-1] in (3, 4):
        t = t[..., :3].permute(0, 3, 1, 2).contiguous()
    elif t.shape[1] in (3, 4):
        t = t[:, :3]
    else:
        raise ValueError(f"Unexpected RGB shape: {tuple(t.shape)}")
    if t.max() > 1.5:
        t = t / 255.0
    return F.interpolate(torch.clamp(t, 0.0, 1.0), size=(image_size, image_size), mode="bilinear", align_corners=False)


def _prep_depth(x, image_size: int, depth_max: float, device):
    src = _to_torch(x, device)
    src_is_int = not src.dtype.is_floating_point
    t = src
    if t.ndim == 4 and t.shape[-1] == 1:
        t = t[..., 0]
    elif t.ndim == 4 and t.shape[1] == 1:
        t = t[:, 0]
    else:
        raise ValueError(f"Unexpected depth shape: {tuple(t.shape)}")
    t = torch.nan_to_num(t.float(), nan=depth_max, posinf=depth_max, neginf=0.0)
    if src_is_int:
        t = torch.clamp(t / 255.0, 0.0, 1.0)
    elif t.max() > 1.5:
        t = torch.clamp(t, 0.05, depth_max)
        t = torch.log1p(t) / torch.log1p(torch.tensor(depth_max, device=device))
    else:
        t = torch.clamp(t, 0.0, 1.0)
    return F.interpolate(t.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False)


def _tensor_stats(name: str, x, env_idx: int = 0) -> str:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    t = t.detach().float().cpu()
    if t.ndim >= 1 and t.shape[0] > env_idx:
        t = t[env_idx]
    nan_n = int(torch.isnan(t).sum().item())
    inf_n = int(torch.isinf(t).sum().item())
    return (
        f"{name}: shape={tuple(x.shape) if hasattr(x, 'shape') else 'n/a'} "
        f"dtype={getattr(x, 'dtype', type(x))} "
        f"min={float(t.min()):.4f} max={float(t.max()):.4f} mean={float(t.mean()):.4f} "
        f"nan={nan_n} inf={inf_n}"
    )


def _depth_branch(x) -> str:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    src_is_int = not t.dtype.is_floating_point
    mx = float(t.float().max().item())
    if src_is_int:
        return "uint8/255"
    if mx > 1.5:
        return "metric log1p(depth_max)"
    return "already normalized [0,1]"


def _debug_vision(
    step: int,
    obs: dict,
    head: torch.Tensor,
    ee: torch.Tensor,
    prop9: torch.Tensor,
    teacher_cmd: torch.Tensor,
    prop_mean: torch.Tensor,
    prop_std: torch.Tensor,
    cmd_mean: torch.Tensor,
    cmd_std: torch.Tensor,
    image_size: int,
    depth_max: float,
    out_dir: str,
    env_idx: int = 0,
) -> None:
    image = obs.get("image", {})
    print(f"[vision-debug] ===== step={step} env_idx={env_idx} =====", flush=True)
    if isinstance(image, dict):
        print(f"[vision-debug] image keys: {sorted(image.keys())}", flush=True)
        for key in ("head_rgb", "head_depth", "ee_rgb", "ee_depth"):
            if key in image:
                print(f"[vision-debug] raw {_tensor_stats(key, image[key], env_idx=env_idx)}", flush=True)
                if "depth" in key:
                    print(f"[vision-debug]   depth branch: {_depth_branch(image[key])}", flush=True)
    else:
        print("[vision-debug] obs['image'] missing or not dict", flush=True)

    proprio = _to_torch(obs["proprio"], head.device).float()
    if proprio.ndim == 1:
        proprio = proprio.unsqueeze(0)
    print(f"[vision-debug] raw proprio dim={proprio.shape[-1]}", flush=True)
    print(f"[vision-debug]   base_lin_vel  {proprio[env_idx, 0:3].detach().cpu().tolist()}", flush=True)
    print(f"[vision-debug]   base_ang_vel  {proprio[env_idx, 3:6].detach().cpu().tolist()}", flush=True)
    print(f"[vision-debug]   vel_cmd(skip) {proprio[env_idx, 6:9].detach().cpu().tolist()}", flush=True)
    print(f"[vision-debug]   proj_gravity  {proprio[env_idx, 9:12].detach().cpu().tolist()}", flush=True)
    print(f"[vision-debug] prop9 {_tensor_stats('prop9', prop9, env_idx=env_idx)}", flush=True)
    print(f"[vision-debug]   prop9 vec     {prop9[env_idx].detach().cpu().tolist()}", flush=True)

    ch_names = ("R", "G", "B", "D")
    for tag, tensor in (("head", head), ("ee", ee)):
        print(f"[vision-debug] proc {tag} {_tensor_stats(tag, tensor, env_idx=env_idx)}", flush=True)
        for ci, cn in enumerate(ch_names):
            c = tensor[env_idx, ci]
            print(
                f"[vision-debug]   {tag} ch-{cn}: min={float(c.min()):.4f} max={float(c.max()):.4f} mean={float(c.mean()):.4f}",
                flush=True,
            )

    head_u8 = (torch.clamp(head, 0.0, 1.0) * 255.0).to(torch.uint8)
    head_rt = head_u8.float() / 255.0
    rt_err = float((head - head_rt).abs().max().item())
    print(f"[vision-debug] uint8 roundtrip head max_abs_err={rt_err:.6f} (expect <=0.004)", flush=True)

    print(f"[vision-debug] norm stats prop_mean={prop_mean.view(-1).detach().cpu().tolist()}", flush=True)
    print(f"[vision-debug] norm stats prop_std ={prop_std.view(-1).detach().cpu().tolist()}", flush=True)
    print(f"[vision-debug] norm stats cmd_mean ={cmd_mean.view(-1).detach().cpu().tolist()}", flush=True)
    print(f"[vision-debug] norm stats cmd_std  ={cmd_std.view(-1).detach().cpu().tolist()}", flush=True)

    prop_n = (prop9 - prop_mean) / prop_std
    cmd_n = (teacher_cmd - cmd_mean) / cmd_std
    print(f"[vision-debug] teacher_cmd raw  {teacher_cmd[env_idx].detach().cpu().tolist()}", flush=True)
    print(f"[vision-debug] teacher_cmd norm {cmd_n[env_idx].detach().cpu().tolist()}", flush=True)
    print(f"[vision-debug] prop9 norm       {prop_n[env_idx].detach().cpu().tolist()}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    try:
        from PIL import Image

        def _save_rgbd(tag: str, tensor: torch.Tensor):
            x = tensor[env_idx].detach().cpu().clamp(0.0, 1.0)
            rgb = (x[:3].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            depth = (x[3].numpy() * 255.0).astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(out_dir, f"step{step:04d}_{tag}_rgb_env{env_idx}.png"))
            Image.fromarray(depth).save(os.path.join(out_dir, f"step{step:04d}_{tag}_depth_env{env_idx}.png"))

        _save_rgbd("head", head)
        _save_rgbd("ee", ee)
        print(f"[vision-debug] saved PNGs to {out_dir}", flush=True)
    except ImportError:
        print("[vision-debug] PIL not installed; skip PNG dump", flush=True)
    print(f"[vision-debug] image_size={image_size} depth_max={depth_max}", flush=True)


def _extract_batch(obs: dict, image_size: int, depth_max: float, device):
    image = obs.get("image")
    if not isinstance(image, dict):
        raise RuntimeError("obs['image'] missing; require head/ee RGB-D")
    required = ("head_rgb", "head_depth", "ee_rgb", "ee_depth")
    miss = [k for k in required if image.get(k) is None]
    if miss:
        raise RuntimeError(f"Missing required camera keys: {miss}")
    head = torch.cat([_prep_rgb(image["head_rgb"], image_size, device), _prep_depth(image["head_depth"], image_size, depth_max, device)], dim=1)
    ee = torch.cat([_prep_rgb(image["ee_rgb"], image_size, device), _prep_depth(image["ee_depth"], image_size, depth_max, device)], dim=1)
    proprio = _to_torch(obs["proprio"], device).float()
    if proprio.ndim == 1:
        proprio = proprio.unsqueeze(0)
    prop9 = torch.cat([proprio[:, 0:3], proprio[:, 3:6], proprio[:, 9:12]], dim=-1)
    return head, ee, prop9


def _read_done(env, num_envs: int, device, terminated, truncated):
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

    def _bt(v):
        if isinstance(v, torch.Tensor):
            return v.to(device=device, dtype=torch.bool).view(-1)[:num_envs]
        return torch.full((num_envs,), bool(v), device=device, dtype=torch.bool)

    term_t = _bt(terminated)
    trunc_t = _bt(truncated)
    if hasattr(unwrapped, "reset_terminated"):
        term_t = _bt(unwrapped.reset_terminated)
    if hasattr(unwrapped, "reset_time_outs"):
        trunc_t = _bt(unwrapped.reset_time_outs)
    reset_buf_t = _bt(getattr(unwrapped, "reset_buf", torch.zeros(num_envs, device=device, dtype=torch.bool)))
    # Use union to avoid missing episode ends when one source lags.
    done_t = reset_buf_t | term_t | trunc_t
    return term_t, trunc_t, done_t


def _termination_thresholds(env) -> tuple[float, float]:
    fall_thresh, x_thresh = 0.25, 3.5
    try:
        cfg = env.unwrapped.cfg if hasattr(env, "unwrapped") else env.cfg
        terms = getattr(cfg, "terminations", None)
        if terms is not None:
            fall_cfg = getattr(terms, "fall", None)
            x_cfg = getattr(terms, "x_reached", None)
            if fall_cfg is not None and isinstance(getattr(fall_cfg, "params", None), dict):
                fall_thresh = float(fall_cfg.params.get("minimum_height", fall_thresh))
            if x_cfg is not None and isinstance(getattr(x_cfg, "params", None), dict):
                x_thresh = float(x_cfg.params.get("x_threshold", x_thresh))
    except Exception:
        pass
    return fall_thresh, x_thresh


def _print_episode_end_logs(
    step: int,
    done_mask: torch.Tensor,
    episode_score: torch.Tensor,
    term_t: torch.Tensor,
    trunc_t: torch.Tensor,
    env,
    elapsed: float,
    fall_thresh: float,
    x_thresh: float,
) -> None:
    if not bool(done_mask.any()):
        return
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    robot = getattr(unwrapped.scene, "articulations", {}).get("robot", None) if hasattr(unwrapped, "scene") else None
    idxs = done_mask.nonzero(as_tuple=False).view(-1).tolist()
    for idx in idxs:
        term = bool(term_t[idx].item())
        trunc = bool(trunc_t[idx].item())
        score = float(episode_score[idx].item())
        print(f"[online-bc] done: env={idx} terminated={int(term)} truncated={int(trunc)}", flush=True)
        if robot is not None:
            pos = robot.data.root_pos_w[idx]
            root_x = float(pos[0].item())
            root_z = float(pos[2].item())
            infer_fall = int(root_z < fall_thresh)
            infer_x = int(root_x > x_thresh)
            print(
                f"[online-bc] infer: env={idx} root_x={root_x:+.3f} (x_thresh={x_thresh:+.3f}) "
                f"root_z={root_z:+.3f} (fall_thresh={fall_thresh:+.3f})",
                flush=True,
            )
            print(
                f"[online-bc] infer terms: env={idx} fall={infer_fall} x_reached={infer_x} time_out={int(trunc)}",
                flush=True,
            )
        print(f"[online-bc] episode_score: env={idx} score={score:.2f}, elapsed_time={elapsed:.2f} seconds", flush=True)


def _import_teacher(module_name: str, class_name: str):
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls()


def _teacher_cmd(teacher, num_envs: int, device):
    cmd = getattr(teacher, "_last_high_level_cmd_batch", None)
    if isinstance(cmd, torch.Tensor) and cmd.ndim == 2 and cmd.shape == (num_envs, 3):
        return cmd.to(device=device, dtype=torch.float32)
    return torch.zeros((num_envs, 3), device=device, dtype=torch.float32)


def _hl_hold_steps(teacher) -> int:
    hold = getattr(teacher, "_hl_cmd_hold_steps", None)
    if hold is not None:
        return max(1, int(hold))
    dt = float(getattr(teacher, "dt", 0.02))
    hz = float(getattr(teacher, "high_level_hz", 10.0))
    return max(1, int(round(1.0 / (hz * dt))))


def _steps_to_env(n: int, step_unit: str, hold: int) -> int:
    n = int(n)
    if step_unit == "inference":
        return n * int(hold)
    return n


def _env_step_to_inference(env_step: int, hold: int) -> int:
    if env_step <= 0:
        return 0
    return (env_step - 1) // int(hold) + 1


def _collate_episodes(episodes, device):
    b = len(episodes)
    lengths = np.asarray([int(ep["prop"].shape[0]) for ep in episodes], dtype=np.int64)
    t_max = int(lengths.max())
    _, channels, img_h, img_w = episodes[0]["head"].shape
    head = torch.zeros((b, t_max, channels, img_h, img_w), device=device, dtype=torch.float32)
    ee = torch.zeros((b, t_max, channels, img_h, img_w), device=device, dtype=torch.float32)
    prop = torch.zeros((b, t_max, 9), device=device, dtype=torch.float32)
    cmd = torch.zeros((b, t_max, 3), device=device, dtype=torch.float32)
    mask = torch.zeros((b, t_max), device=device, dtype=torch.bool)
    done = torch.zeros((b, t_max), device=device, dtype=torch.bool)
    for i, ep in enumerate(episodes):
        n = int(lengths[i])
        head[i, :n] = torch.from_numpy(ep["head"]).to(device=device, dtype=torch.float32) / 255.0
        ee[i, :n] = torch.from_numpy(ep["ee"]).to(device=device, dtype=torch.float32) / 255.0
        prop[i, :n] = torch.from_numpy(ep["prop"]).to(device=device, dtype=torch.float32)
        cmd[i, :n] = torch.from_numpy(ep["cmd"]).to(device=device, dtype=torch.float32)
        done[i, :n] = torch.from_numpy(ep["done"]).to(device=device, dtype=torch.bool)
        mask[i, :n] = True
    return head, ee, prop, cmd, mask, done


def train_step_full_episode(model, opt, episodes, prop_mean, prop_std, cmd_mean, cmd_std, device):
    if len(episodes) == 0:
        return 0.0
    head, ee, prop, cmd, mask, done = _collate_episodes(episodes, device)
    b, t_max = prop.shape[:2]
    prop_n = (prop - prop_mean.view(1, 1, -1)) / prop_std.view(1, 1, -1)
    cmd_n = (cmd - cmd_mean.view(1, 1, -1)) / cmd_std.view(1, 1, -1)

    h = torch.zeros((1, b, model.hidden_dim), device=device, dtype=torch.float32)
    per_step_losses = []
    for k in range(t_max):
        step_mask = mask[:, k]
        if not bool(step_mask.any()):
            continue
        pred, h = model.forward_step(head[:, k], ee[:, k], prop_n[:, k], h)
        step_loss = F.smooth_l1_loss(pred, cmd_n[:, k], reduction="none").mean(dim=-1)
        per_step_losses.append(step_loss * step_mask.float())
        if k + 1 < t_max and bool(done[:, k].any()):
            h[:, done[:, k], :] = 0.0

    if len(per_step_losses) == 0:
        return 0.0
    loss = torch.stack(per_step_losses, dim=1).sum() / mask.float().sum().clamp_min(1.0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())


def _print_score_stats(
    step,
    total_steps,
    completed_scores,
    completed_scores_valid,
    episode_score,
    done_this_step,
    num_envs,
    updates,
    loss,
    elapsed,
    replay_size: int = 0,
    step_unit: str = "env",
    hold_steps: int = 1,
    total_steps_input: int | None = None,
):
    n_done = len(completed_scores)
    running_mean = float(episode_score.mean().item())
    running_max = float(episode_score.max().item())
    eps_per_env = n_done / max(num_envs, 1)
    if n_done > 0:
        done_avg = float(np.mean(completed_scores))
        done_min = float(np.min(completed_scores))
        done_max = float(np.max(completed_scores))
        done_str = f"finished_avg={done_avg:.3f} (min={done_min:.3f} max={done_max:.3f} n={n_done})"
    else:
        done_str = "finished_avg=n/a (no episode ended yet)"
    n_valid = len(completed_scores_valid)
    if n_valid > 0:
        valid_avg = float(np.mean(completed_scores_valid))
        valid_str = f" scored_episodes={n_valid} scored_avg={valid_avg:.3f} replay={replay_size}"
    else:
        valid_str = f" scored_episodes=0 replay={replay_size}"
    if step_unit == "inference":
        inf_step = _env_step_to_inference(step, hold_steps)
        inf_total = int(total_steps_input if total_steps_input is not None else total_steps)
        step_str = (
            f"infer_step={inf_step:6d}/{inf_total} "
            f"(env_step={step:6d}/{total_steps}, hold={hold_steps})"
        )
    else:
        step_str = f"env_step={step:6d}/{total_steps}"
    print(
        f"[online-bc] {step_str} done_episodes={n_done} ({eps_per_env:.2f}/env) done_now={done_this_step} "
        f"running_score={running_mean:.3f} (max={running_max:.3f}) {done_str}{valid_str} "
        f"updates={updates} loss={loss:.6f} elapsed={elapsed:.1f}s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Online BC with per-env episode replay and full-episode training.")
    parser.add_argument("--task", type=str, default="ATEC-TaskD-B2Piper")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--total_steps", type=int, default=30000)
    parser.add_argument(
        "--step_unit",
        type=str,
        default="inference",
        choices=("env", "inference"),
        help="Unit for --total_steps and other *_steps args. "
        "'inference' = teacher high-level steps (~10Hz); 'env' = env loop steps (~50Hz).",
    )
    parser.add_argument("--teacher_module", type=str, default="demo.solution")
    parser.add_argument("--teacher_class", type=str, default="AlgSolution")
    parser.add_argument("--buffer_len", type=int, default=64)
    parser.add_argument("--chunk_len", type=int, default=24)
    parser.add_argument("--train_every_steps", type=int, default=5)
    parser.add_argument("--batch_envs", type=int, default=8, help="Episode batch size sampled from replay per update.")
    parser.add_argument("--warmup_steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--depth_max", type=float, default=5.0)
    parser.add_argument("--disable_lidar", action="store_true", default=True)
    parser.add_argument("--log_every_steps", type=int, default=200)
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--teacher_only_rollout",
        action="store_true",
        default=False,
        help="Teacher-only mode: run teacher.predicts -> env.step -> done/reset only (no BC extract/train/save).",
    )
    parser.add_argument("--video", action="store_true", default=False, help="Record a debug video during online training.")
    parser.add_argument("--video_length", type=int, default=1500, help="Recorded debug video length in env steps.")
    parser.add_argument("--video_dir", type=str, default=None, help="Optional output dir for debug videos.")
    parser.add_argument(
        "--debug_vision",
        action="store_true",
        default=False,
        help="Print/save vision preprocessing debug (raw obs, 4ch tensor, normalization) then exit.",
    )
    parser.add_argument("--debug_vision_steps", type=int, default=3, help="Env steps to dump when --debug_vision.")
    parser.add_argument("--debug_vision_env", type=int, default=0, help="Env index for vision debug dumps.")
    parser.add_argument("--debug_vision_dir", type=str, default=None, help="PNG output dir for --debug_vision.")
    parser.add_argument(
        "--replay_capacity",
        type=int,
        default=256,
        help="Max number of scored full episodes kept in global replay.",
    )
    parser.add_argument(
        "--min_episode_score",
        type=float,
        default=0.0,
        help="Only store/train episodes with episode_score strictly greater than this value.",
    )
    parser.add_argument(
        "--episodes_per_env",
        type=int,
        default=None,
        help="Deprecated alias for --replay_capacity (kept for backward compatibility).",
    )
    parser.add_argument(
        "--min_episode_len",
        type=int,
        default=8,
        help="Only store/train completed episodes with at least this many steps.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    replay_capacity = int(args.replay_capacity)
    if args.episodes_per_env is not None:
        replay_capacity = int(args.episodes_per_env)
        print("[online-bc] warning: --episodes_per_env is deprecated; use --replay_capacity.", flush=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: E402
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
    import atec_rl_lab.tasks  # noqa: F401, E402
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

    device = torch.device(args.device)
    num_envs = int(args.num_envs)
    step_unit = str(args.step_unit)
    total_steps_input = int(args.total_steps)

    teacher = _import_teacher(args.teacher_module, args.teacher_class)
    print(f"[online-bc] teacher={args.teacher_module}.{args.teacher_class}", flush=True)
    if hasattr(teacher, "set_device"):
        teacher.set_device(args.device)
    if num_envs > 1 and not hasattr(teacher, "reset_env_batch"):
        raise RuntimeError("Teacher from demo/test.py must support reset_env_batch for num_envs>1.")
    hl_hold_steps = _hl_hold_steps(teacher)
    total_steps = _steps_to_env(total_steps_input, step_unit, hl_hold_steps)
    warmup_steps = _steps_to_env(int(args.warmup_steps), step_unit, hl_hold_steps)
    train_every_steps = _steps_to_env(int(args.train_every_steps), step_unit, hl_hold_steps)
    log_every_steps = _steps_to_env(int(args.log_every_steps), step_unit, hl_hold_steps)
    save_every_steps = _steps_to_env(int(args.save_every_steps), step_unit, hl_hold_steps)
    print(
        f"[online-bc] step_unit={step_unit} hl_hold_steps={hl_hold_steps} "
        f"total={total_steps_input} -> env_total={total_steps}",
        flush=True,
    )

    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.abspath(os.path.join("logs", "bc_online", args.task, stamp))
    else:
        out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[online-bc] output dir: {out_dir}", flush=True)
    debug_vision = bool(args.debug_vision)
    if debug_vision:
        debug_env_steps = _steps_to_env(int(args.debug_vision_steps), step_unit, hl_hold_steps)
        total_steps = min(total_steps, debug_env_steps)
        teacher_only = False
        print(
            f"[online-bc] debug_vision=ON steps={int(args.debug_vision_steps)} ({step_unit}) "
            f"-> env_steps={total_steps} env_idx={int(args.debug_vision_env)}",
            flush=True,
        )
    debug_vision_dir = (
        os.path.abspath(args.debug_vision_dir)
        if args.debug_vision_dir
        else os.path.join(out_dir, "vision_debug")
    )

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs, use_fabric=not getattr(args, "disable_fabric", False))
    if args.disable_lidar:
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "lidar_sensor"):
            env_cfg.scene.lidar_sensor = None
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "extero"):
            env_cfg.observations.extero = None
        print("[online-bc] LiDAR disabled; camera observations required.", flush=True)

    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args.video:
        if args.video_dir is None:
            video_dir = os.path.abspath(os.path.join(out_dir, "playback"))
        else:
            video_dir = os.path.abspath(args.video_dir)
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda s: s == 0,
            video_length=int(args.video_length),
            disable_logger=True,
        )
        print(
            f"[online-bc] video enabled: folder={video_dir} length={int(args.video_length)} steps",
            flush=True,
        )
    if hasattr(teacher, "bind_env") and "TaskD" in str(args.task):
        teacher.bind_env(env)
    fall_thresh, x_thresh = _termination_thresholds(env)

    teacher_only = bool(args.teacher_only_rollout) and not debug_vision
    if teacher_only:
        print("[online-bc] teacher_only_rollout=ON (skip extract/train/checkpoint).", flush=True)
        model = None
        prop_mean = None
        prop_std = None
        cmd_mean = None
        cmd_std = None
        opt = None
        replay = None
    else:
        model = TaskDBCGRUPolicy().to(device)
        prop_mean = torch.zeros((1, 9), device=device, dtype=torch.float32)
        prop_std = torch.ones((1, 9), device=device, dtype=torch.float32)
        cmd_mean = torch.zeros((1, 3), device=device, dtype=torch.float32)
        cmd_std = torch.ones((1, 3), device=device, dtype=torch.float32)
        warm = os.path.join("logs", "bc", "taskd_run1", "best.pt")
        if os.path.exists(warm):
            ckpt = torch.load(warm, map_location=device)
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
                st = ckpt.get("stats")
                if st is not None:
                    prop_mean = torch.tensor(st["proprio_mean"], device=device, dtype=torch.float32)
                    prop_std = torch.tensor(st["proprio_std"], device=device, dtype=torch.float32)
                    cmd_mean = torch.tensor(st["cmd_mean"], device=device, dtype=torch.float32)
                    cmd_std = torch.tensor(st["cmd_std"], device=device, dtype=torch.float32)
                print(f"[online-bc] warm-start from {warm}", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        replay = ScoredEpisodeReplayBuffer(
            num_envs=num_envs,
            replay_capacity=replay_capacity,
            min_episode_score=float(args.min_episode_score),
        )
        print(
            "[online-bc] training mode=scored_full_episode_replay "
            f"(replay_capacity={replay_capacity} min_episode_score>{float(args.min_episode_score):.3f} "
            f"min_episode_len={int(args.min_episode_len)}).",
            flush=True,
        )
        print(
            f"[online-bc] note: chunk_len={int(args.chunk_len)} / buffer_len={int(args.buffer_len)} are ignored in this mode.",
            flush=True,
        )

    obs, _ = env.reset()
    if hasattr(teacher, "reset"):
        teacher.reset(task=args.task)
    if hasattr(teacher, "bind_env") and "TaskD" in str(args.task):
        teacher.bind_env(env)

    step = 0
    updates = 0
    last_loss = 0.0
    episode_score = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    prev_done_t = torch.zeros((num_envs,), device=device, dtype=torch.bool)
    completed_scores = []
    completed_scores_valid = []
    t0 = time.time()
    while simulation_app.is_running() and step < total_steps:
        step += 1
        resp = teacher.predicts(obs, float(episode_score.mean().item()))
        if bool(resp.get("giveup", False)):
            print("[online-bc] teacher giveup=True stop.", flush=True)
            break
        action_tensor = resp.get("action_tensor")
        if isinstance(action_tensor, torch.Tensor):
            actions = action_tensor.to(device=device, dtype=torch.float32)
        else:
            actions = torch.as_tensor(resp["action"], device=device, dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if actions.shape[0] != num_envs:
            raise RuntimeError(f"Action batch mismatch: got {actions.shape[0]} expect {num_envs}")

        teacher_cmd = _teacher_cmd(teacher, num_envs, device)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        reward_t = reward if isinstance(reward, torch.Tensor) else torch.as_tensor(reward, device=device)
        reward_t = reward_t.view(-1).to(device=device, dtype=torch.float32)
        sim_dt = info.get("Step_dt", 1.0) if isinstance(info, dict) else 1.0
        if isinstance(sim_dt, torch.Tensor):
            sim_dt = float(sim_dt.view(-1)[0].item())
        sim_dt = float(sim_dt)
        episode_score = episode_score + reward_t / max(sim_dt, 1e-8)

        term_t, trunc_t, done_t = _read_done(env, num_envs=num_envs, device=device, terminated=terminated, truncated=truncated)
        newly_done = done_t & (~prev_done_t)
        if bool(newly_done.any()):
            vals = episode_score[newly_done].detach().cpu().numpy().tolist()
            completed_scores.extend(float(v) for v in vals)
            completed_scores_valid.extend(float(v) for v in vals if float(v) > float(args.min_episode_score))
            _print_episode_end_logs(
                step=step,
                done_mask=newly_done,
                episode_score=episode_score,
                term_t=term_t,
                trunc_t=trunc_t,
                env=env,
                elapsed=(time.time() - t0),
                fall_thresh=fall_thresh,
                x_thresh=x_thresh,
            )

        if not teacher_only:
            head, ee, prop9 = _extract_batch(obs, image_size=int(args.image_size), depth_max=float(args.depth_max), device=device)
            if debug_vision:
                _debug_vision(
                    step=step,
                    obs=obs,
                    head=head,
                    ee=ee,
                    prop9=prop9,
                    teacher_cmd=teacher_cmd,
                    prop_mean=prop_mean,
                    prop_std=prop_std,
                    cmd_mean=cmd_mean,
                    cmd_std=cmd_std,
                    image_size=int(args.image_size),
                    depth_max=float(args.depth_max),
                    out_dir=debug_vision_dir,
                    env_idx=int(args.debug_vision_env),
                )
            replay.add_batch(
                head_u8=(torch.clamp(head, 0.0, 1.0) * 255.0).to(torch.uint8).detach().cpu().numpy(),
                ee_u8=(torch.clamp(ee, 0.0, 1.0) * 255.0).to(torch.uint8).detach().cpu().numpy(),
                prop=prop9.detach().cpu().numpy().astype(np.float32),
                cmd=teacher_cmd.detach().cpu().numpy().astype(np.float32),
                done=done_t.detach().cpu().numpy(),
            )
            if bool(newly_done.any()):
                before = len(replay)
                replay.finalize_done(
                    newly_done.detach().cpu().numpy(),
                    episode_score.detach().cpu().numpy(),
                    min_len=int(args.min_episode_len),
                )
                added = len(replay) - before
                if added > 0:
                    print(
                        f"[online-bc] replay +{added} scored episodes (size={len(replay)}/{replay.replay_capacity})",
                        flush=True,
                    )

        if bool(done_t.any()):
            if hasattr(teacher, "reset_env_batch"):
                teacher.reset_env_batch(done_t)
                setattr(teacher, "_hl_cmd_force_refresh", True)
            episode_score[done_t] = 0.0

        if (not teacher_only) and step >= warmup_steps and step % train_every_steps == 0:
            episodes = replay.sample_episodes(batch_size=int(args.batch_envs), min_len=int(args.min_episode_len))
            if len(episodes) > 0:
                last_loss = train_step_full_episode(
                    model, opt, episodes, prop_mean, prop_std, cmd_mean, cmd_std, device
                )
                updates += 1

        if step % log_every_steps == 0 or step == total_steps:
            _print_score_stats(
                step=step,
                total_steps=total_steps,
                completed_scores=completed_scores,
                completed_scores_valid=completed_scores_valid,
                episode_score=episode_score,
                done_this_step=int(newly_done.sum().item()),
                num_envs=num_envs,
                updates=updates,
                loss=last_loss,
                elapsed=time.time() - t0,
                replay_size=len(replay) if replay is not None else 0,
                step_unit=step_unit,
                hold_steps=hl_hold_steps,
                total_steps_input=total_steps_input,
            )

        if (not teacher_only) and step % save_every_steps == 0:
            path = os.path.join(out_dir, f"online_step_{step:07d}.pt")
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "stats": {
                        "proprio_mean": prop_mean.detach().cpu().numpy().tolist(),
                        "proprio_std": prop_std.detach().cpu().numpy().tolist(),
                        "cmd_mean": cmd_mean.detach().cpu().numpy().tolist(),
                        "cmd_std": cmd_std.detach().cpu().numpy().tolist(),
                    },
                    "args": vars(args),
                },
                path,
            )
            print(f"[online-bc] saved: {path}", flush=True)

        prev_done_t = done_t.clone()
        obs = next_obs

    if not teacher_only:
        final_path = os.path.join(out_dir, "online_last.pt")
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "stats": {
                    "proprio_mean": prop_mean.detach().cpu().numpy().tolist(),
                    "proprio_std": prop_std.detach().cpu().numpy().tolist(),
                    "cmd_mean": cmd_mean.detach().cpu().numpy().tolist(),
                    "cmd_std": cmd_std.detach().cpu().numpy().tolist(),
                },
                "args": vars(args),
            },
            final_path,
        )
        print(
            f"[online-bc] done. step={step} updates={updates} last_loss={last_loss:.6f} "
            f"replay={len(replay)} scored_added={replay.total_scored_added} discarded={replay.total_discarded} "
            f"ckpt={final_path}",
            flush=True,
        )
    else:
        print(f"[online-bc] done (teacher-only). step={step}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
