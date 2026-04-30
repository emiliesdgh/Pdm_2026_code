"""
Script to regroup all the global variables used in the project
this is to avoid redundancy of redefining the same variables in different files,
and to have a single source of truth for these variables.
"""

import numpy as np

class GlobalVariables:

    def __init__(self, frame):
        ### === Global for fingers === ###
        self.FINGERS ={
            "name": ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"],
            "tip_idx": [4, 8, 12, 16, 20],
            "dip_idx": [3, 7, 11, 15, 19],
            "pip_idx": [2, 6, 10, 14, 18],
            "base_idx": [1, 5, 9, 13, 17]
        }

        self.WRIST_idx = 0

        ### === Camera information === ###
        ## NEED to find the actial focal length of the camera for accurate real-world coordinate conversion
        # will need to do a camera calibration process to get the actual focal length and other intrinsic parameters for accurate real-world coordinate conversion
        self.FOCAL_LENGTH = 1  # Assuming normalized coordinates where focal length is 1
        ### === Camera information === ###
        self.frame = frame
        self.H = frame.shape[0] # Height
        self.W = frame.shape[1] # Width
        self.D = frame.shape[2] # Depth

        self.FOV = 1.32  # radians
        self.FOCAL_LENGTH = self.W / (2 * np.tan(self.FOV / 2))  # Focal length of the camera
        self.centerFrame = (self.W // 2, self.H // 2)

        self.camera_view = np.array([0, 0, -1])

        ### === CV2 drawing variables === ###
        # self.TEXT_FLIPPED = True

        ### === Thresholds for gesture recognition === ###
        self.FINGER_STATE_TH = 1     # At least 1 finger state change
        self.ORIENTATION_TH = 1      # If the orientation direction is different
        self.MOVE_TH = 0.05          # Threshold for displacement