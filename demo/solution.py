import os
from enum import Enum

import numpy as np
import open3d as o3d
import torch

from demo.utils import approach_dustbin


class Status(Enum):
    SEARCH = 1
    LOCK = 2
    PICK = 3
    STAND = 4
    GO_BIN = 5


class AlgSolution:
    ACTION_SCALE = 0.5
    _SQUAT_STEPS = 100
    _PICK_ARM_STEPS = 25
    _STAND_STEPS = 80
    _BIN_ARRIVE_DIST = 1.0
    _GO_BIN_ALIGN_RAD = 0.35
    _GO_BIN_WZ_GAIN = 0.8
    _GO_BIN_VX = 1.5
    EE_BODY_NAME_CANDIDATES = ("gripper_base", "piper_gripper_base")
    ARM_JOINT_NAME_CANDIDATES = (
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"],
    )

    def calculate_velocity(
        self, target_x, target_y, max_vx, max_vy, max_wz, k_v=0.5, k_w=1.0, *, lock_on_arrive=True
    ):
        """
        target_x, target_y: 目标相对坐标
        max_vx, max_vy, max_wz: 速度上限
        k_v: 线性速度增益
        k_w: 角速度增益
        """
        # 1. 计算距离和角度
        dist = np.sqrt(target_x ** 2 + target_y ** 2)
        angle_to_target = np.arctan2(target_y, target_x)

        # 2. 计算目标速度 (随着距离减小，速度线性下降)
        # 使用 min(k*dist, max_v) 实现限幅
        vx = np.clip(k_v * target_x, -max_vx, max_vx)
        vy = np.clip(k_v * target_y, -max_vy, max_vy)

        # 3. 角速度控制 (假设机器人车头朝向为X轴)
        # 仅当目标距离较远时才大幅旋转，近距离时减小旋转以防震荡
        wz = np.clip(k_w * angle_to_target, -max_wz, max_wz)

        # 4. 捡垃圾：仅 SEARCH 阶段触发 LOCK；GO_BIN 不触发
        if lock_on_arrive and dist < 0.4:
            self.status = Status.LOCK
            self.get_down = True
            self.start_get_down_idx = self.cur_idx
            self.cmd_max_vx = 0.0
            self.cmd_max_vy = 0.0
            self.cmd_max_wz = 0.0
            return 0.0, 0.0, 0.0
        # 与 SEARCH 相同：写死 cmd，供策略 obs 里的 velocity_commands 使用
        self.cmd_max_vy = 0
        self.cmd_max_wz = 0.8 * -target_x
        self.cmd_max_vx = 1.5 * 1
        return vx, vy, wz

    def _obs_depth(self, obs, key: str) -> np.ndarray | None:
        image = obs.get("image")
        if not image or key not in image:
            return None
        return image[key][0].to(self.device).cpu().numpy()

    def _obs_extero(self, obs) -> np.ndarray | None:
        extero = obs.get("extero")
        if extero is None:
            return None
        return extero.to(self.device).cpu().numpy()[0]

    def _begin_search_cycle(self) -> None:
        self.status = Status.SEARCH
        self.get_down = False
        self.start_get_down_idx = None
        self.start_pick_idx = None
        self.start_stand_idx = None
        self.v_list = [0.0 for _ in range(8)]
        self.cmd_max_vx = 1.0
        self.cmd_max_vy = 0.0
        self.cmd_max_wz = 1.0
        self._last_bin_bearing = None
        self._last_bin_dist = None
        print("[TaskB] bin reached -> SEARCH (loop)", flush=True)

    def _update_go_bin_cmd(self, obs) -> None:
        fused = approach_dustbin(
            self._obs_extero(obs),
            self._obs_depth(obs, "head_depth"),
            self.K,
        )
        bearing = fused.get("bearing")
        dist = fused.get("dist")
        self._last_bin_bearing = bearing
        self._last_bin_dist = dist

        if dist is not None and float(dist) < self._BIN_ARRIVE_DIST:
            self._begin_search_cycle()
            return

        if bearing is None:
            self.cmd_max_vx = 0.0
            self.cmd_max_vy = 0.0
            self.cmd_max_wz = 0.0
            return

        bearing_f = float(bearing)
        if abs(bearing_f) > self._GO_BIN_ALIGN_RAD:
            # 先原地对准 LiDAR bearing，不要 vx=1.5 硬冲
            self.cmd_max_vx = 0.0
            self.cmd_max_vy = 0.0
            self.cmd_max_wz = float(
                np.clip(self._GO_BIN_WZ_GAIN * bearing_f, -0.8, 0.8)
            )
            return

        self.cmd_max_vx = self._GO_BIN_VX
        self.cmd_max_vy = 0.0
        self.cmd_max_wz = float(np.clip(0.4 * bearing_f, -0.4, 0.4))

    def __init__(self):
        policy_path = os.path.dirname(os.path.abspath(__file__)) + '/policy.pt'
        print(policy_path)
        self.device = 'cuda'

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.leg_action_dim = 12
        self.arm_action_dim = 8

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        self.train_to_env_action_scale = torch.tensor(
            [0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5],
            device=self.device, dtype=torch.float32
        ).view(1, -1)

        self.env_to_train_action_scale = torch.tensor(
            [4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0],
            device=self.device, dtype=torch.float32
        ).view(1, -1)

        self.arm_default_action = torch.zeros((1, self.arm_action_dim), device=self.device, dtype=torch.float32)

        self.cmd_max_vx = 1.0
        self.cmd_max_vy = 0.5
        self.cmd_max_wz = 1.0
        self.status = Status.SEARCH
        self.start_stand_idx = None
        self._last_bin_bearing = None
        self._last_bin_dist = None
        # ==========================================
        # 虚拟里程计 (用于在世界坐标系下展示点云移动)
        # ==========================================
        self.vis_x = 0.0
        self.vis_y = 0.0
        self.vis_yaw = 0.0

        # 传感器安装高度 (机器人 base 离地高度)
        self.sensor_height = 0.6

        self.first_render = True
        self.debug_printed = False

        # 2. 根据你的 Isaac Lab 配置重建雷达光束的方向
        channels = 16  # 16线
        num_horizontal_rays = 360  # 360度，每度1个点

        # 俯仰角 (Vertical FOV: -20 到 20 度)
        pitch_angles = np.linspace(np.radians(-20.0), np.radians(20.0), channels)
        # 偏航角 (Horizontal FOV: -180 到 180 度)
        yaw_angles = np.linspace(np.radians(-180.0), np.radians(180.0), num_horizontal_rays)

        # 生成网格
        pitch_grid, yaw_grid = np.meshgrid(pitch_angles, yaw_angles, indexing="ij")
        self.pitch_grid = pitch_grid.flatten()
        self.yaw_grid = yaw_grid.flatten()
        self.inited = False
        self.vis = None
        self.pcd = None
        self.coord_frame = None
        self.key = None
        self.auto = False
        self.v_list = [0 for _ in range(8)]
        self.get_down = False
        self.start_get_down_idx = None
        self.start_pick_idx = None
        self.start_pose = None
        self.cur_idx = 0
        self.K = np.array([[458.12, 0, 320], [0, 458.12, 240], [0, 0, 1]])


    def init(self):
        # ==========================================
        # 初始化 Open3D 非阻塞可视化器
        # ==========================================
        if self.inited:
            return
        self.auto = True
        self.start_pick_idx = None
        if os.environ.get("DISABLE_O3D_VIS", "0") == "1":
            self.inited = True
            return
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Isaac Lab LiDAR Viewer", width=1024, height=768)

        def space_callback(vis):
            self.key = None
            return False

        def w_key_callback(vis):
            self.key = 'w'
            print('w')
            return False

        def a_key_callback(vis):
            self.key = 'a'
            return False

        def s_key_callback(vis):
            self.key = 's'
            return False

        def d_key_callback(vis):
            self.key = 'd'
            return False

        def i_key_callback(vis):
            self.auto = True
            self.status = Status.SEARCH
            return False

        def k_key_callback(vis):
            self.auto = False
            self.v_list = [0 for _ in range(8)]
            return False

        def n_key_callback(vis):
            self.get_down = True
            return False

        def m_key_callback(vis):
            self.get_down = False
            return False

        def key4_callback(vis):
            self.v_list[0] += 0.2
            return False

        def key6_callback(vis):
            self.v_list[0] -= 0.2
            return False

        def key8_callback(vis):
            self.v_list[1] += 0.2
            return False

        def key2_callback(vis):
            self.v_list[1] -= 0.2
            return False

        def o_key_callback(vis):
            self.v_list[2] += 0.2
            return False

        def p_key_callback(vis):
            self.v_list[2] -= 0.2
            return False

        self.vis.register_key_callback(ord(' '), space_callback)
        self.vis.register_key_callback(ord('W'), w_key_callback)
        self.vis.register_key_callback(ord('A'), a_key_callback)
        self.vis.register_key_callback(ord('S'), s_key_callback)
        self.vis.register_key_callback(ord('D'), d_key_callback)
        self.vis.register_key_callback(ord('I'), i_key_callback)
        self.vis.register_key_callback(ord('K'), k_key_callback)
        self.vis.register_key_callback(ord('N'), n_key_callback)
        self.vis.register_key_callback(ord('M'), m_key_callback)
        self.vis.register_key_callback(ord('G'), key4_callback)
        self.vis.register_key_callback(ord('J'), key6_callback)
        self.vis.register_key_callback(ord('Y'), key8_callback)
        self.vis.register_key_callback(ord('H'), key2_callback)

        self.vis.register_key_callback(ord('O'), o_key_callback)
        self.vis.register_key_callback(ord('P'), p_key_callback)

        # 创建全局 PointCloud 和坐标系几何体
        self.pcd = o3d.geometry.PointCloud()
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        # 将几何体添加到渲染器 (此时点云是空的)
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.coord_frame)
        self.cur_idx = 0
        self.inited = True

    def __del__(self):
        """安全销毁窗口，防止退出时崩溃"""
        try:
            self.vis.destroy_window()
        except Exception:
            pass

    def reset(self, **kwargs):
        del kwargs
        self.vis_x = 0.0
        self.vis_y = 0.0
        self.vis_yaw = 0.0
        self.first_render = True
        self.debug_printed = False
        self.status = Status.SEARCH
        self.get_down = False
        self.start_get_down_idx = None
        self.start_pick_idx = None
        self.start_stand_idx = None
        self.v_list = [0.0 for _ in range(8)]
        self.cmd_max_vx = 1.0
        self.cmd_max_vy = 0.0
        self.cmd_max_wz = 1.0
        self._last_bin_bearing = None
        self._last_bin_dist = None
        self.cur_idx = 0

    def _resolve_joint_ids(self, candidates: tuple[list[str], ...]) -> list[int]:
        for names in candidates:
            try:
                ids, found_names = self.robot.find_joints(names)
                if len(ids) == len(names):
                    if candidates is self.ARM_JOINT_NAME_CANDIDATES:
                        self.arm_joint_names = list(found_names)
                    return list(ids)
            except ValueError:
                continue
        raise ValueError("Cannot resolve required joints.")

    def _resolve_ee_body_name(self) -> str:
        for name in self.EE_BODY_NAME_CANDIDATES:
            try:
                body_ids, _ = self.robot.find_bodies(name)
                if len(body_ids) == 1: return name
            except ValueError:
                continue
        raise ValueError("Cannot resolve EE body.")

    def _ensure_cartesian_targets(self):
        self.cartesian_ctrl.reset()

    def _compute_arm_overlay_action(self) -> torch.Tensor:
        self._ensure_cartesian_targets()
        arm_jpos_des = self.cartesian_ctrl.compute_base(self.ee_pos_target_b, self.ee_quat_target_b)
        full_target = self.robot.data.joint_pos.clone()
        full_target[:, self.arm_ids] = arm_jpos_des
        full_target[:, self.gripper_ids] = self.gripper_open_pos.repeat(full_target.shape[0], 1)
        return (full_target - self.default_joint_pos) / self.ACTION_SCALE

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        b = proprio.shape[0]
        vx_cmd, vy_cmd, yaw_cmd = 0.0, 0.0, 0.0
        if self.auto:
            vx_cmd = self.cmd_max_vx
            yaw_cmd = self.cmd_max_wz
            vy_cmd = self.cmd_max_vy
        else:
            if self.key == 'w':
                vx_cmd = 1
            elif self.key == 's':
                vx_cmd = -1
            if self.key == 'a':
                yaw_cmd = 1.0
            elif self.key == 'd':
                yaw_cmd = -1.0
            if self.key == 'q':
                vy_cmd = self.cmd_max_vy
            elif self.key == 'e':
                vy_cmd = -self.cmd_max_vy

        return torch.tensor([[vx_cmd, vy_cmd, yaw_cmd]], device=proprio.device, dtype=proprio.dtype).repeat(b, 1)

    def _extract_policy_obs(self, obs, action_dim) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        idx = 3
        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 6
        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3
        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]

        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)
        velocity_commands = self._get_velocity_commands(proprio)

        return torch.cat([
            base_ang_vel * 0.25, projected_gravity, velocity_commands,
            joint_pos_leg, joint_vel_leg * 0.05, actions_train_leg,
        ], dim=-1)

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)
        return action_env

    def find_start_point_optimized(self, head_depth):
        # 1. 计算每一行中相邻列的差值
        # diffs[y, x] = head_depth[y, x+1] - head_depth[y, x]
        # 我们关注的是当前像素比左侧像素小的情况，即 diff < -0.1

        # 获取所有行的 x=1 到 639 的数据与 x=0 到 638 的数据差值
        diffs = head_depth[:, 1:] - head_depth[:, :-1]

        # 2. 寻找满足条件的点 (diff < -0.1)
        # 得到一个布尔矩阵，形状为 (480, 639)
        mask = diffs < -0.05

        # 3. 从底部向上遍历行 (range(479, -1, -1))
        for y in range(479, -1, -1):
            # 找到该行中第一个为 True 的索引
            indices = np.where(mask[y])[0]
            if indices.size > 0:
                # 找到第一个符合条件的 x (注意因为我们用了 [:, 1:]，所以 x 需要 +1)
                x = indices[0] + 1
                return x, y
        return None

    def detect_ground_ransac(self, points, distance_threshold=0.05, ransac_n=3, num_iterations=1000):
        """
        使用RANSAC检测地面平面
        points: (N, 3) 点云
        distance_threshold: 点到平面的距离阈值(米)
        ransac_n: 每次采样点数(平面需要3)
        num_iterations: 迭代次数
        返回: (ground_mask, plane_model)
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 执行平面分割
        plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)

        # 提取平面内点
        ground_mask = np.zeros(len(points), dtype=bool)
        ground_mask[inliers] = True

        # 平面模型: [a, b, c, d] 满足 a*x + b*y + c*z + d = 0
        return ground_mask, plane_model

    def depth_to_point_cloud(self, depth_map):
        height, width = depth_map.shape
        # 1. 创建像素网格
        i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
        # 2. 将像素坐标转换为相机坐标系下的点
        K = self.K
        z = -depth_map
        x = -(i - K[0, 2]) * z / K[0, 0]
        y = (j - K[1, 2]) * z / K[1, 1]

        # 3. 堆叠成 (N, 3) 的点云数组
        point_cloud = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        # 过滤掉无效深度值 (例如远端截断值)
        valid_mask = (depth_map > 0.05) & (depth_map < 50.0)
        return point_cloud[valid_mask.reshape(-1)]

    def transform_ground_to_zero(self, points, plane_model):
        """
        points: (N, 3) 原始点云
        plane_model: (a, b, c, d) 平面参数
        """
        a, b, c, d = plane_model
        normal = np.array([a, b, c])

        # 1. 计算旋转矩阵 (Rotation Matrix)
        # 目标是将 normal 旋转到 [0, 0, 1]
        target_normal = np.array([0, 0, 1])

        # 使用罗德里格斯旋转公式或求旋转轴和角
        v = np.cross(normal, target_normal)
        s = np.linalg.norm(v)
        c_val = np.dot(normal, target_normal)

        # 反对称矩阵
        v_skew = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])

        # 旋转矩阵 R
        R = np.eye(3) + v_skew + np.dot(v_skew, v_skew) * ((1 - c_val) / (s ** 2 + 1e-9))

        # 2. 计算平移向量
        # 平面上距离原点最近的点是 P0 = -d * normal
        p0 = -d * normal

        # 3. 构建变换矩阵 T (从世界到地面坐标系的变换)
        # 新坐标 P' = R * (P - p0)
        # 这一步使地面点旋转到水平且 p0 移到原点
        points_transformed = (R @ (points - p0).T).T

        return points_transformed, R, p0

    def cluster_euclidean_open3d(self, points, eps=0.1, min_points=10, max_points=10000):
        """
        使用Open3D的欧几里得聚类（速度快，适合大点云）

        Args:
            points: (N, 3) 点云
            eps: 聚类半径
            min_points: 最小点数
            max_points: 最大点数

        Returns:
            cluster_labels: 每个点的聚类标签
            n_clusters: 聚类数量
        """
        # 转换为Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # 欧几里得聚类
        cluster_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"Open3D聚类: 找到 {n_clusters} 个物体")
        return cluster_labels, n_clusters

    def find_target_by_depth(self, depth):
        points = self.depth_to_point_cloud(depth.squeeze())
        ground_mask, plane_model = self.detect_ground_ransac(points, distance_threshold=0.03)
        points_flat, R, p0 = self.transform_ground_to_zero(points, plane_model)
        others = points_flat[~ground_mask]
        if len(others) == 0:
            return None, None
        labels, n_clusters = self.cluster_euclidean_open3d(others)
        min_dist = None
        target = None
        # 1. 筛选并计算每个聚类的中心距离
        for label in range(n_clusters):
            cur_points = others[labels == label]
            z_max = np.max(cur_points[:, 2])
            if z_max > 0.4:
                continue
            # 计算该簇的几何中心
            centroid = np.mean(cur_points, axis=0)
            dist_to_origin = np.linalg.norm(centroid)
            if min_dist is None or dist_to_origin < min_dist:
                min_dist = dist_to_origin
                target = centroid
        return target, min_dist

    def predicts(self, obs, current_score):
        del current_score
        self.init()
        if self.vis is not None:
            self.vis.poll_events()
            self.vis.update_renderer()
        self.cur_idx += 1

        if self.status == Status.SEARCH:
            if self.cur_idx % 4 == 0:
                head_depth = self._obs_depth(obs, "head_depth")
                target, min_dist = (None, None)
                if head_depth is not None:
                    target, min_dist = self.find_target_by_depth(head_depth)
                if target is None:
                    ee_depth = self._obs_depth(obs, "ee_depth")
                    if ee_depth is not None:
                        target, min_dist = self.find_target_by_depth(ee_depth)
                if target is not None:
                    self.calculate_velocity(target[0], target[1], 1, 0.5, 1)
                print(f"[SEARCH] target={target}, min_dist={min_dist}", flush=True)

        elif self.status == Status.PICK and self.start_pick_idx is not None:
            if self.cur_idx < self.start_pick_idx + 10:
                self.v_list[1] += 0.1
            if self.cur_idx < self.start_pick_idx + self._PICK_ARM_STEPS:
                self.v_list[1] += 0.1
                self.v_list[2] -= 0.06
            elif self.cur_idx >= self.start_pick_idx + self._PICK_ARM_STEPS:
                self.status = Status.STAND
                self.get_down = False
                self.start_stand_idx = self.cur_idx
                self.v_list = [0.0 for _ in range(8)]
                self.cmd_max_vx = 0.0
                self.cmd_max_vy = 0.0
                self.cmd_max_wz = 0.0
                print("[TaskB] PICK done -> STAND", flush=True)

        elif self.status == Status.STAND and self.start_stand_idx is not None:
            if self.cur_idx >= self.start_stand_idx + self._STAND_STEPS:
                self.status = Status.GO_BIN
                print("[TaskB] STAND done -> GO_BIN (LiDAR yaw + depth range)", flush=True)

        elif self.status == Status.GO_BIN:
            if self.cur_idx % 4 == 0:
                self._update_go_bin_cmd(obs)
                print(
                    f"[GO_BIN] bearing={self._last_bin_bearing} dist={self._last_bin_dist}",
                    flush=True,
                )

        proprio = obs["proprio"].to(self.device)

        # ==========================================
        # 策略推理
        # ==========================================
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        policy_obs = self._extract_policy_obs(obs, action_dim)

        with torch.inference_mode():
            action_train = self.policy(policy_obs)

        action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        if action_train.ndim == 1: action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        # return {'action': action_env.cpu().numpy().tolist(), 'giveup': False}

        proprio = obs['proprio']
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        action = [0 for _ in range(action_dim)]
        use_policy_legs = self.status in (Status.SEARCH, Status.STAND, Status.GO_BIN) or (
            self.status == Status.LOCK and not self.get_down
        )
        if use_policy_legs:
            aa = action_env.cpu().numpy().tolist()
            action[:action_dim] = aa[0]
        else:
            self.cmd_max_vx = 0.0
            self.cmd_max_vy = 0.0
            self.cmd_max_wz = 0.0
            squat_pose = [
                0.0, 0.5, -1,
                0.0, 0.5, -1,
                0.0, 0.5, -1,
                0.0, 0.5, -1,
            ]
            if (
                self.start_get_down_idx is not None
                and self.cur_idx == self.start_get_down_idx + self._SQUAT_STEPS
            ):
                self.status = Status.PICK
                self.start_pick_idx = self.cur_idx
                print("[TaskB] squat done -> PICK", flush=True)
            action[:12] = squat_pose
        if self.status == Status.PICK:
            action[12:20] = self.v_list
        else:
            action[12:20] = [0.0 for _ in range(8)]
        return {'action': action, 'giveup': False}
