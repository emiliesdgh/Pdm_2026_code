"""
Script to determine if the hand gesture is a motion or stationary
to be able to know if iconic gestures are static (ex: thumbs up) or dynamic (ex: swipe left) 
"""

import numpy as np
from collections import deque

class TemporalGestureManager:
    def __init__(self, window_size=15):
        self.history_dist = deque(maxlen=window_size)
        self.gesture_history = deque(maxlen=window_size)
        self.wrist_history = deque(maxlen=window_size)
        self.window_size = window_size

    def update(self, hand_landmarks, finger_states, hand_orientation, hand_position):
        """
        Processes new frame data and returns if there is a temporal gesture detected.
        Returns: (is_moving: bool, motion_type: str)
        """
        wrist = hand_landmarks.landmark[0]
        wrist_pos = np.array([wrist.x, wrist.y])

        # Store the information
        self.gesture_history.append((finger_states, hand_orientation))
        self.wrist_history.append(wrist_pos)

        return self.analyze_motion()
    
    def analyze_motion(self):
        if len(self.gesture_history) < self.window_size:
            return False, "Detecting..."  # Need enough frames to detect a trend

        # 1. Detect if there is a change in finger states or orientation
        prefix = list(self.gesture_history)[:self.window_size//2]
        suffix = list(self.gesture_history)[-self.window_size//2:]

        finger_state_change = self.finger_change_score(prefix, suffix)
        orientation_change = self.orientation_change_score(prefix, suffix)
        
        # 2. Analyze the detailed trajectory (Speed, Path, Flips)
        displacement, path_length, speed_str, flips_x, flips_y = self.analyze_trajectory()

        # --- Thresholds --- (tune these based on your camera resolution/distance)
        FINGER_STATE_TH = 1     # At least 1 finger state change
        ORIENTATION_TH = 1      # If the orientation direction is different
        MOVE_TH = 0.05          # Threshold for displacement
        
        magnitude = np.linalg.norm(displacement)

        # Note: We check 'path_length' here too! 
        # If someone waves, they might end up exactly where they started (magnitude = 0), 
        # but the path length will be high!
        is_moving = (
            finger_state_change >= FINGER_STATE_TH or 
            orientation_change >= ORIENTATION_TH or 
            magnitude >= MOVE_TH or 
            path_length >= (MOVE_TH * 1.5) 
        )

        if not is_moving:
            return False, "Stationary"
        
        # --- Determine type of motion ---
        motion_type = self.classify_motion(
            finger_state_change, 
            orientation_change, 
            displacement,
            speed_str,
            flips_x,
            flips_y
        )

        return True, motion_type
    
    def analyze_trajectory(self):
        """
        Calculates advanced spatial metrics: Speed, exact path length, and directional flips (waving).
        """
        history = list(self.wrist_history)
        displacement = history[-1] - history[0]
        
        path_length = 0.0
        x_dirs = []
        y_dirs = []

        # Analyze frame-by-frame changes
        for i in range(1, len(history)):
            diff = history[i] - history[i-1]
            path_length += np.linalg.norm(diff)
            
            # Record the direction (+ or -) of movement, ignoring micro-jitters
            jitter_th = 0.005
            if abs(diff[0]) > jitter_th: x_dirs.append(np.sign(diff[0]))
            if abs(diff[1]) > jitter_th: y_dirs.append(np.sign(diff[1]))

        # Helper to count how many times the hand changed direction
        def count_flips(dir_list):
            return sum(1 for i in range(1, len(dir_list)) if dir_list[i] != dir_list[i-1])

        flips_x = count_flips(x_dirs)
        flips_y = count_flips(y_dirs)

        # Calculate Average Speed (Path length per frame)
        avg_speed = path_length / self.window_size
        if avg_speed > 0.03: speed_str = "Fast"
        elif avg_speed > 0.01: speed_str = "Moderate"
        else: speed_str = "Slow"

        return displacement, path_length, speed_str, flips_x, flips_y

    def finger_change_score(self, prefix, suffix):
        prefix_states = np.array([state for state, _ in prefix])
        suffix_states = np.array([state for state, _ in suffix])

        prefix_majority = (prefix_states.mean(axis=0) > 0.5).astype(int)
        suffix_majority = (suffix_states.mean(axis=0) > 0.5).astype(int)

        score = np.sum(np.abs(prefix_majority != suffix_majority))
        return score

    def orientation_change_score(self, prefix, suffix):
        prefix_dirs = [d for _, d in prefix]
        suffix_dirs = [d for _, d in suffix]

        prefix_majority = max(set(prefix_dirs), key=prefix_dirs.count)
        suffix_majority = max(set(suffix_dirs), key=suffix_dirs.count)

        return 1 if prefix_majority != suffix_majority else 0
    
    def classify_motion(self, finger_change, orientation_change, displacement, speed_str, flips_x, flips_y):
        dx, dy = displacement
        magnitude = np.linalg.norm(displacement)

        # -- 1. Wave Detection (Oscillation) --
        # If the direction flips back and forth 2 or more times, it's a wave
        if flips_x >= 2 or flips_y >= 2:
            return f"{speed_str} Waving"

        # -- 2. Articulation --
        if finger_change >= 4:
            return f"{speed_str} Hand Open/Close"
        if finger_change >= 1 and finger_change < 4:
            return f"{speed_str} Bending Fingers"
        
        # -- 3. Rotation --
        if orientation_change: 
            return f"{speed_str} Hand Rotation"
        
        # -- 4. Swipe / Translation (8-way directional mapping) --
        if magnitude > 0.05: 
            # Calculate angle of displacement (-180 to 180 degrees)
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Map angle to 8 points of the compass
            ## Directions are different due to webcam mirroring
            ## to adjust when using robot camera

            if -22.5 <= angle < 22.5: dir_str = "Right"
            elif 22.5 <= angle < 67.5: dir_str = "Down-Right"
            elif 67.5 <= angle < 112.5: dir_str = "Down"
            elif 112.5 <= angle < 157.5: dir_str = "Down-Left"
            elif angle >= 157.5 or angle < -157.5: dir_str = "Left"
            elif -157.5 <= angle < -112.5: dir_str = "Up-Left"
            elif -112.5 <= angle < -67.5: dir_str = "Up"
            elif -67.5 <= angle < -22.5: dir_str = "Up-Right"
            else: dir_str = "Unknown Direction"

            # Note: Because the camera is mirrored, depending on how your 
            # final UI works, you may need to flip "Left" and "Right" here.
            
            # if the string ends by Left
            if dir_str.endswith("Left"):
                dir_str = dir_str.replace("Left", "Right")
            elif dir_str.endswith("Right"):
                dir_str = dir_str.replace("Right", "Left")
            
            return f"{speed_str} Swipe {dir_str}"
            
        return f"{speed_str} Unknown Motion"