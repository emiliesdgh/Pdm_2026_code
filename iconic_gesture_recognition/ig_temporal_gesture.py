"""
Script to determine if the hand gesture is a motion or stationnary
to be able to know if iconic gestures are static (ex: thumbs up) or dynamic (ex: swipe left) 
"""

import numpy as np
from collections import deque

class TemporalGestureManager:
    def __init__(self, window_size=15):
        # Stores the distance between index tip and thumb tip over time
        self.history_dist = deque(maxlen=window_size)
        # Stores the detected static gestures to help smooth results
        self.gesture_history = deque(maxlen=window_size)
        self.wrist_history = deque(maxlen=window_size)
        self.window_size = window_size

    def update(self, hand_landmarks, finger_states, hand_orientation):
        """
        Processes new frame data and returns if there is a temporal gesture detected.
        to detect that, we want to see if there is a change in the finger states and hand orientation
        True --> motion detected
        False --> stationary

        returns: (is_moving: bool, motion_type: str)
        """
        wrist = hand_landmarks.landmark[0]
        wrist_pos = np.array([wrist.x, wrist.y])

        # store the information
        self.gesture_history.append((finger_states, hand_orientation))
        self.wrist_history.append(wrist_pos)

        return self.analyze_motion()

    
    def analyze_motion(self):
        if len(self.gesture_history) < self.window_size:
            return False, "Detecting..."  # Need enough frames to detect a trend

        # if len(self.history_dist) < self.window_size:
        #     return None
        
        # 1. detect if there is a change in finger states that corresponds to a temporal gesture
        prefix = list(self.gesture_history)[:self.window_size//2]   # instead of 6 and -6 ? because where tf it came from this value
        suffix = list(self.gesture_history)[-self.window_size//2:]

        finger_state_change = self.finger_change_score(prefix, suffix)
        orientation_change = self.orientation_change_score(prefix, suffix)
        mouvement_vector = self.wrist_movement()

        # --- Thresholds --- (to be tuned based on experimentation)
        # if too sensitive --> increase thresholds
        # if missing gestures --> decrease thresholds
        FINGER_STATE_TH = 1     # At least 1 finger state change
        ORIENTATION_TH = 30     # At least 30 degrees change in orientation
        MOVE_TH = 0.05          # At least 5% of the frame size movement

        # if finger_state_change >= FINGER_STATE_TH:
        #     return True
        
        # if orientation_change >= ORIENTATION_TH:
        #     return True
        
        # return False

        is_moving = (
            finger_state_change >= FINGER_STATE_TH or 
            orientation_change >= ORIENTATION_TH or 
            np.linalg.norm(mouvement_vector) >= MOVE_TH
        )

        if not is_moving:
            return False, "Stationary"
        
        # --- Determine type of motion ---
        motion_type = self.classify_motion(
            finger_state_change, 
            orientation_change, 
            mouvement_vector
        )

        return True, motion_type
    
    def finger_change_score(self, prefix, suffix):
        # Count how many fingers changed state from prefix to suffix
        prefix_states = np.array([state for state, _ in prefix])
        suffix_states = np.array([state for state, _ in suffix])

        # majority vote per finger
        prefix_majority = (prefix_states.mean(axis=0) > 0.5).astype(int)
        suffix_majority = (suffix_states.mean(axis=0) > 0.5).astype(int)

        # We can use a simple sum of absolute differences as a score
        # score = np.sum(np.abs(prefix_majority - suffix_majority)) # ?
        score = np.sum(np.abs(prefix_majority != suffix_majority))

        return score

    def orientation_change_score(self, prefix, suffix):
        # # Calculate the average orientation in prefix and suffix
        # prefix_orientations = np.array([orientation for _, orientation in prefix])
        # suffix_orientations = np.array([orientation for _, orientation in suffix])

        # prefix_avg = np.mean(prefix_orientations, axis=0)
        # suffix_avg = np.mean(suffix_orientations, axis=0)

        # # Calculate the angle difference (assuming orientations are angles in degrees)
        # angle_diff = np.abs(prefix_avg - suffix_avg)

        # return angle_diff
        prefix_angles = [angle for _, angle in prefix]
        suffix_angles = [angle for _, angle in suffix]

        prefix_mean = np.mean(prefix_angles)
        suffix_mean = np.mean(suffix_angles)

        return self.angle_diff(prefix_mean, suffix_mean)
    
    def angle_diff(self, angle1, angle2):
        """Compute smallest difference between two angles (degrees)"""
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)
    
    
    def wrist_movement(self):
        if len(self.wrist_history) < self.window_size:
            return np.array([0, 0])  # Not enough data to compute movement
        
        start = self.wrist_history[0]
        end = self.wrist_history[-1]

        return end - start
    
    def classify_motion(self, finger_change, orientation_change, movement_vector):
        dx, dy = movement_vector
        magnitude = np.linalg.norm(movement_vector)

        # -- gesture change --
        if finger_change >= 2: # arbitrary threshold for multiple fingers changing
            return "hand_open_close"
        
        # -- rotation --
        if orientation_change > 20: # arbitrary threshold for significant rotation
            return "hand_rotation"
        
        # -- translation / swip detection --
        if magnitude > 0.05: # arbitrary threshold for significant movement
            if abs(dx) > abs(dy):
                # check for correct direction based on camera feed mirroring
                return "swipe_left" if dx > 0 else "swipe_right"
            else:
                return "swipe_down" if dy > 0 else "swipe_up"
            
        return "unknown_motion"
        



