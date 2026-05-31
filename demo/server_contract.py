"""Platform /step observation contract (demo/server.py).

Train with --platform_depth_train should render the same depth layout the server
passes into your upload code on each step.
"""

# server.py step(): np.float32 reshaped to (1, H, W, 1)
PLATFORM_DEPTH_H = 480
PLATFORM_DEPTH_W = 640
