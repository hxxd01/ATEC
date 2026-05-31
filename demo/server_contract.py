"""Platform /step observation contract (demo/server.py).

Train with --platform_depth_train should render the same depth layout the server
passes into your upload code on each step. Policy input defaults to 24x32 (img_h x img_w).
"""

# server.py step(): np.float32 reshaped to (1, H, W, 1)
PLATFORM_DEPTH_H = 480
PLATFORM_DEPTH_W = 640
POLICY_IMG_H = 24
POLICY_IMG_W = 32
