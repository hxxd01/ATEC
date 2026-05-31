"""Platform observation contract from demo/server.py /step (single source for train/deploy).

server.py reshapes uploads as:
  head_depth / video_depth / ee_depth: float32 (1, 480, 640, 1)
  head_rgb / video_rgb / ee_rgb: uint8 (1, 480, 640, 3)
  proprio: float32 (1, -1)
"""

SERVER_DEPTH_H = 480
SERVER_DEPTH_W = 640
