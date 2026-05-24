import argparse
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_pickle_stream(path: str):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def _resolve_image_keys(step_record: dict) -> dict[str, str]:
    imgs = step_record.get("images", {})
    if not isinstance(imgs, dict) or not imgs:
        return {}

    keys = list(imgs.keys())
    out = {}
    for k in keys:
        lk = k.lower()
        if "head" in lk and "rgb" in lk:
            out["head_rgb"] = k
        elif "head" in lk and "depth" in lk:
            out["head_depth"] = k
        elif "ee" in lk and "rgb" in lk:
            out["ee_rgb"] = k
        elif "ee" in lk and "depth" in lk:
            out["ee_depth"] = k
    return out


def _build_episodes_from_trajectory(
    path: str, require_images: bool = True, progress_every_steps: int = 200
) -> tuple[list[dict], dict]:
    per_env_buffers: dict[int, list[dict]] = defaultdict(list)
    episodes: list[dict] = []
    image_key_map: dict[str, str] | None = None
    meta = {}
    step_count = 0

    for rec in _load_pickle_stream(path):
        if not isinstance(rec, dict):
            continue
        rtype = rec.get("type", "")
        if rtype == "meta":
            meta = rec
            continue
        if rtype != "step":
            continue

        step_count += 1
        if progress_every_steps > 0 and step_count % progress_every_steps == 0:
            print(
                f"[bc] loading pkl... parsed_step_records={step_count} finished_episodes={len(episodes)}",
                flush=True,
            )
        env_indices = rec.get("env_indices", None)
        if env_indices is None:
            n = int(np.asarray(rec["high_level_cmd"]).shape[0])
            env_indices = np.arange(n, dtype=np.int64)
        env_indices = np.asarray(env_indices, dtype=np.int64)

        cmd = np.asarray(rec["high_level_cmd"], dtype=np.float32)
        base_lin = np.asarray(rec["base_lin_vel"], dtype=np.float32)
        base_ang = np.asarray(rec["base_ang_vel"], dtype=np.float32)
        grav = np.asarray(rec["projected_gravity"], dtype=np.float32)
        done = np.asarray(rec["done"]).astype(np.bool_)

        images = rec.get("images", {})
        if image_key_map is None and isinstance(images, dict) and images:
            image_key_map = _resolve_image_keys(rec)

        for i, env_id in enumerate(env_indices):
            sample = {
                "proprio": np.concatenate([base_lin[i], base_ang[i], grav[i]], axis=0).astype(np.float32),
                "cmd": cmd[i].astype(np.float32),
            }
            if image_key_map and isinstance(images, dict):
                sample["images"] = {
                    "head_rgb": np.asarray(images[image_key_map["head_rgb"]][i]),
                    "head_depth": np.asarray(images[image_key_map["head_depth"]][i]),
                    "ee_rgb": np.asarray(images[image_key_map["ee_rgb"]][i]),
                    "ee_depth": np.asarray(images[image_key_map["ee_depth"]][i]),
                }
            per_env_buffers[int(env_id)].append(sample)

            if done[i]:
                ep_samples = per_env_buffers[int(env_id)]
                if len(ep_samples) > 0:
                    ep = {
                        "proprio": np.stack([s["proprio"] for s in ep_samples], axis=0),  # [T, 9]
                        "cmd": np.stack([s["cmd"] for s in ep_samples], axis=0),  # [T, 3]
                    }
                    if "images" in ep_samples[0]:
                        ep["head_rgb"] = np.stack([s["images"]["head_rgb"] for s in ep_samples], axis=0)
                        ep["head_depth"] = np.stack([s["images"]["head_depth"] for s in ep_samples], axis=0)
                        ep["ee_rgb"] = np.stack([s["images"]["ee_rgb"] for s in ep_samples], axis=0)
                        ep["ee_depth"] = np.stack([s["images"]["ee_depth"] for s in ep_samples], axis=0)
                    episodes.append(ep)
                per_env_buffers[int(env_id)] = []

    if require_images:
        if not episodes:
            raise RuntimeError("No finished episodes found in trajectory file.")
        if "head_rgb" not in episodes[0]:
            raise RuntimeError(
                "Dataset does not contain images. Re-collect with --store_images (and preferably --store_uint8_images=False)."
            )

    print(f"[bc] loaded steps={step_count}, finished_episodes={len(episodes)} from {path}", flush=True)
    return episodes, meta


def _compute_norm_stats(episodes: list[dict]) -> dict[str, np.ndarray]:
    proprios = np.concatenate([ep["proprio"] for ep in episodes], axis=0)
    cmds = np.concatenate([ep["cmd"] for ep in episodes], axis=0)
    stats = {
        "proprio_mean": proprios.mean(axis=0, keepdims=True).astype(np.float32),
        "proprio_std": np.clip(proprios.std(axis=0, keepdims=True), 1e-3, None).astype(np.float32),
        "cmd_mean": cmds.mean(axis=0, keepdims=True).astype(np.float32),
        "cmd_std": np.clip(cmds.std(axis=0, keepdims=True), 1e-3, None).astype(np.float32),
    }
    return stats


class TaskDBCDataset(Dataset):
    def __init__(
        self,
        episodes: list[dict],
        seq_len: int,
        stride: int,
        stats: dict[str, np.ndarray],
        image_size: int = 96,
        depth_max: float = 5.0,
    ):
        self.episodes = episodes
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.stats = stats
        self.image_size = int(image_size)
        self.depth_max = float(depth_max)
        self.index: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(episodes):
            t = int(ep["proprio"].shape[0])
            if t < self.seq_len:
                continue
            for s in range(0, t - self.seq_len + 1, self.stride):
                self.index.append((ep_idx, s))

    def __len__(self):
        return len(self.index)

    def _prep_rgb(self, x: np.ndarray) -> torch.Tensor:
        arr = np.asarray(x)
        # Be tolerant to singleton dimensions produced by different camera backends:
        # e.g. (H,W,3,1) / (1,H,W,3) / (H,W,1,3) -> (H,W,3)
        if arr.ndim > 3:
            arr = np.squeeze(arr)
        if arr.ndim == 4:
            if arr.shape[-1] == 1 and arr.shape[-2] == 3:
                arr = np.squeeze(arr, axis=-1)
            elif arr.shape[-2] == 1 and arr.shape[-1] == 3:
                arr = np.squeeze(arr, axis=-2)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB with shape (H,W,3), got {tuple(arr.shape)}")

        t = torch.from_numpy(arr).float()
        if t.max() > 1.5:
            t = t / 255.0
        t = t.permute(2, 0, 1).contiguous()  # [3,H,W]
        return t

    def _prep_depth(self, x: np.ndarray) -> torch.Tensor:
        src_dtype = x.dtype
        arr = np.asarray(x)
        if arr.ndim > 3:
            arr = np.squeeze(arr)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"Expected depth with shape (H,W) or (H,W,1), got {tuple(arr.shape)}")

        t = torch.from_numpy(arr).float()
        t = torch.nan_to_num(t, nan=self.depth_max, posinf=self.depth_max, neginf=0.0)
        if np.issubdtype(src_dtype, np.integer):
            # Stored as uint8 image-like depth (already quantized).
            t = t / 255.0
            t = torch.clamp(t, 0.0, 1.0)
        elif t.max() > 1.5:
            # Metric depth (meters) path.
            t = torch.clamp(t, 0.05, self.depth_max)
            t = torch.log1p(t)
            t = t / torch.log1p(torch.tensor(self.depth_max))
        else:
            # Already normalized float depth path.
            t = torch.clamp(t, 0.0, 1.0)
        return t.unsqueeze(0).contiguous()  # [1,H,W]

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def __getitem__(self, idx: int):
        ep_idx, s = self.index[idx]
        ep = self.episodes[ep_idx]
        e = s + self.seq_len

        proprio = ep["proprio"][s:e]  # [T,9]
        cmd = ep["cmd"][s:e]  # [T,3]
        proprio = (proprio - self.stats["proprio_mean"]) / self.stats["proprio_std"]
        cmd = (cmd - self.stats["cmd_mean"]) / self.stats["cmd_std"]

        head = []
        ee = []
        for t in range(self.seq_len):
            hrgb = self._resize(self._prep_rgb(ep["head_rgb"][s + t]))
            hdep = self._resize(self._prep_depth(ep["head_depth"][s + t]))
            ergb = self._resize(self._prep_rgb(ep["ee_rgb"][s + t]))
            edep = self._resize(self._prep_depth(ep["ee_depth"][s + t]))
            head.append(torch.cat([hrgb, hdep], dim=0))  # [4,H,W]
            ee.append(torch.cat([ergb, edep], dim=0))  # [4,H,W]

        return (
            torch.stack(head, dim=0).float(),  # [T,4,H,W]
            torch.stack(ee, dim=0).float(),  # [T,4,H,W]
            torch.from_numpy(proprio).float(),  # [T,9]
            torch.from_numpy(cmd).float(),  # [T,3]
        )


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
        y = self.net(x).flatten(1)
        return self.proj(y)


class TaskDBCGRUPolicy(nn.Module):
    def __init__(self, proprio_dim: int = 9, cmd_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.head_encoder = ConvEncoder(in_ch=4, out_dim=128)
        self.ee_encoder = ConvEncoder(in_ch=4, out_dim=128)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(128 + 128 + 64, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cmd_dim),
        )

    def forward(self, head_img: torch.Tensor, ee_img: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        # head_img/ee_img: [B,T,4,H,W], proprio: [B,T,9]
        b, t = head_img.shape[:2]
        h_feat = self.head_encoder(head_img.view(b * t, *head_img.shape[2:]))
        e_feat = self.ee_encoder(ee_img.view(b * t, *ee_img.shape[2:]))
        p_feat = self.proprio_mlp(proprio.view(b * t, -1))
        fused = self.fuse(torch.cat([h_feat, e_feat, p_feat], dim=-1)).view(b, t, -1)
        out, _ = self.gru(fused)
        return self.head(out)  # [B,T,3], normalized cmd


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for head, ee, proprio, cmd in loader:
            head = head.to(device)
            ee = ee.to(device)
            proprio = proprio.to(device)
            cmd = cmd.to(device)
            pred = model(head, ee, proprio)
            loss = F.smooth_l1_loss(pred, cmd)
            total += float(loss.item()) * head.shape[0]
            n += head.shape[0]
    return total / max(1, n)


def main():
    parser = argparse.ArgumentParser("Train TaskD BC policy (vision+proprio->GRU->cmd)")
    parser.add_argument("--data", type=str, required=True, help="Path to trajectories.pkl")
    parser.add_argument("--save_dir", type=str, default="logs/bc/taskd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--depth_max", type=float, default=5.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_episodes", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--load_progress_every_steps",
        type=int,
        default=200,
        help="Print pkl loading progress every N step records (0 disables).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    episodes, meta = _build_episodes_from_trajectory(
        args.data,
        require_images=True,
        progress_every_steps=args.load_progress_every_steps,
    )
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    if len(episodes) < 2:
        raise RuntimeError("Need at least 2 episodes to split train/val.")

    n_val = max(1, int(len(episodes) * args.val_ratio))
    n_train = len(episodes) - n_val
    train_eps = episodes[:n_train]
    val_eps = episodes[n_train:]

    stats = _compute_norm_stats(train_eps)
    np.savez(
        os.path.join(args.save_dir, "norm_stats.npz"),
        proprio_mean=stats["proprio_mean"],
        proprio_std=stats["proprio_std"],
        cmd_mean=stats["cmd_mean"],
        cmd_std=stats["cmd_std"],
    )

    train_ds = TaskDBCDataset(
        train_eps,
        seq_len=args.seq_len,
        stride=args.stride,
        stats=stats,
        image_size=args.image_size,
        depth_max=args.depth_max,
    )
    val_ds = TaskDBCDataset(
        val_eps,
        seq_len=args.seq_len,
        stride=args.stride,
        stats=stats,
        image_size=args.image_size,
        depth_max=args.depth_max,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"No training chunks created (train={len(train_ds)}, val={len(val_ds)}). "
            "Try reducing --seq_len or --stride."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = TaskDBCGRUPolicy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    print(
        f"[bc] episodes train/val={len(train_eps)}/{len(val_eps)}  chunks train/val={len(train_ds)}/{len(val_ds)}",
        flush=True,
    )
    print(f"[bc] meta: {meta}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for head, ee, proprio, cmd in train_loader:
            head = head.to(device, non_blocking=True)
            ee = ee.to(device, non_blocking=True)
            proprio = proprio.to(device, non_blocking=True)
            cmd = cmd.to(device, non_blocking=True)

            pred = model(head, ee, proprio)
            loss = F.smooth_l1_loss(pred, cmd)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item()) * head.shape[0]
            n += head.shape[0]

        train_loss = running / max(1, n)
        val_loss = evaluate(model, val_loader, device)
        print(f"[bc] epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "args": vars(args),
            "stats": {k: v.tolist() for k, v in stats.items()},
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))

    print(f"[bc] done. best_val={best_val:.6f} save_dir={args.save_dir}", flush=True)


if __name__ == "__main__":
    main()

