"""
Script to determine if the hand gesture is a motion or stationary
to be able to know if iconic gestures are static (ex: thumbs up) or dynamic (ex: swipe left) 
"""

import numpy as np
from collections import deque


class TemporalGestureManager:
    def __init__(self, global_vars, window_size=15):
        self.history_dist = deque(maxlen=window_size)
        self.gesture_history = deque(maxlen=window_size)
        self.wrist_history = deque(maxlen=window_size)

        self.window_size = window_size
        self.global_vars = global_vars

    def update(self, hand_landmarks, finger_states, finger_contacts, hand_orientation, hand_position):
        """
        Processes new frame data and returns if there is a temporal gesture detected.
        Returns: (is_moving: bool, motion_type: str)
        """
        wrist = hand_landmarks.landmark[0]
        wrist_pos = np.array([wrist.x, wrist.y])
        # wrist_pos = np.array([self.global_vars.WRIST.x, self.global_vars.WRIST.y])

        # Store the information
        self.gesture_history.append((finger_states, finger_contacts, hand_orientation))
        self.wrist_history.append(wrist_pos)

        return self.analyze_motion()
    
    def analyze_motion(self):
        if len(self.gesture_history) < self.window_size:
            return False, "Detecting...", "None"  # Need enough frames to detect a trend

        # 1. Detect if there is a change in finger states or orientation
        prefix = list(self.gesture_history)[:self.window_size//2]
        suffix = list(self.gesture_history)[-self.window_size//2:]

        finger_state_change, finger_bending_direction = self.finger_change_score(prefix, suffix)
        orientation_change = self.orientation_change_score(prefix, suffix)
        
        # 2. Analyze the detailed trajectory (Speed, Path, Flips)
        displacement, path_length, speed_str, flips_x, flips_y = self.analyze_trajectory()

        
        magnitude = np.linalg.norm(displacement)

        # Note: We check 'path_length' here too! 
        # If someone waves, they might end up exactly where they started (magnitude = 0), 
        # but the path length will be high!
        is_moving = (
            finger_state_change >= self.global_vars.FINGER_STATE_TH or 
            orientation_change >= self.global_vars.ORIENTATION_TH or 
            magnitude >= self.global_vars.MOVE_TH or 
            path_length >= (self.global_vars.MOVE_TH * 1.5) 
        )

        if not is_moving:
            return False, "Stationary", "None"
        
        # --- Determine type of motion ---
        spatial_motion, articulation = self.classify_motion(
            finger_state_change, 
            finger_bending_direction,
            orientation_change, 
            displacement,
            speed_str,
            flips_x,
            flips_y
        )

        # return True, motion_type, articulation
        return True, spatial_motion, articulation
    
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
        # refine the thresholds for speed categories based on empirical observations !!
        avg_speed = path_length / self.window_size
        if avg_speed > 0.03: speed_str = "Fast"
        elif avg_speed > 0.01: speed_str = "Moderate"
        else: speed_str = "Slow"

        return displacement, path_length, speed_str, flips_x, flips_y

    def finger_change_score_fixe(self, prefix, suffix):
        prefix_states = np.array([state for state, _ in prefix])
        suffix_states = np.array([state for state, _ in suffix])

        prefix_majority = (prefix_states.mean(axis=0) > 0.5).astype(int)
        suffix_majority = (suffix_states.mean(axis=0) > 0.5).astype(int)

        score = np.sum(np.abs(prefix_majority != suffix_majority))
        return score

    def finger_change_score(self, prefix, suffix):
        """
        Calculates how many fingers changed state AND the direction of the change (Opening vs Closing).
        1 = Extended, -1 = Folded.
        """
        prefix_states = np.array([state for state, _, _ in prefix])
        suffix_states = np.array([state for state, _, _ in suffix])
        suffix_contacts = np.array([contacts for _, contacts, _ in suffix])


        # Get the average state in the first half vs second half of the window
        prefix_mean = prefix_states.mean(axis=0)
        suffix_mean = suffix_states.mean(axis=0)

        # Detect direction of change
        # A finger closes if it goes from > 0 (extended) to < 0 (folded)
        closing_fingers = np.sum((prefix_mean > 0) & (suffix_mean < 0))
        # A finger opens if it goes from < 0 (folded) to > 0 (extended)
        opening_fingers = np.sum((prefix_mean < 0) & (suffix_mean > 0))

        is_prefix_open = np.sum(prefix_mean > 0) >= 3   # at least 3 fingers extended
        # --- THE REFINED PINCH MATH ---
        # suffix_contacts represents [Index, Middle, Ring, Pinky]
        latest_contacts = suffix_contacts[-1]
        
        # Check if the Index finger specifically is touching the thumb
        is_index_pinching = latest_contacts[0] == 1

        # Or if it is a full grab (2 or more fingers touching the thumb)
        total_fingers_touching = np.sum(latest_contacts == 1)
        is_true_pinch = is_index_pinching or (total_fingers_touching >= 2)

        total_changes = closing_fingers + opening_fingers

        # Determine the primary articulation direction
        if is_prefix_open and is_true_pinch:
            direction = "Pinching"
        elif closing_fingers > opening_fingers:
            direction = "Closing"# (Grabbing)"
        elif opening_fingers > closing_fingers:
            direction = "Opening"# (Releasing)"
        else:
            direction = "Shifting"

        return total_changes, direction
    
    def orientation_change_score(self, prefix, suffix):
        prefix_dirs = [d for _, _, d in prefix]
        suffix_dirs = [d for _, _, d in suffix]

        prefix_majority = max(set(prefix_dirs), key=prefix_dirs.count)
        suffix_majority = max(set(suffix_dirs), key=suffix_dirs.count)

        return 1 if prefix_majority != suffix_majority else 0
    
    def classify_motion(self, finger_change, finger_dir, orientation_change, displacement, speed_str, flips_x, flips_y):
        dx, dy = displacement
        magnitude = np.linalg.norm(displacement)

        spatial_motion = "Stationary"
        articulation = "None"

        # -- 1. Evaluate Articulation --
        if finger_change >= 4:
            # articulation = f"Dynamic Hand {finger_dir}"     # e.g., "Dynamic Hand Closing (Grabbing)"
            articulation = f"{finger_dir}"     # e.g., "Dynamic Hand Closing (Grabbing)"
        elif finger_change >= 2: 
            # articulation = f"Dynamic Fingers {finger_dir}"  # e.g., "Dynamic Fingers Opening (Releasing)"
            articulation = f"{finger_dir}"  # e.g., "Dynamic Fingers Opening (Releasing)"
        else:
            articulation = "Static Fingers (No articulation change)"

        # -- 2. Wave Detection (Oscillation) --
        # If the direction flips back and forth 2 or more times, it's a wave
        if flips_x >= 2 or flips_y >= 2:
            spatial_motion = f"{speed_str} Oscillating / Waving"
    
        # -- 3. Check for wrist Rotations --
        elif orientation_change: 
            spatial_motion = f"{speed_str} Hand Rotation"

        # -- 4. Start with checking for significant spatial movements --
        # check if possible several ==> append and have more than 1 in spatial motion
        elif magnitude > 0.05: 
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
            
            spatial_motion = f"{speed_str} Linear Translation towards {dir_str}"
        
        return spatial_motion, articulation
    
    # def classify_motion(self, finger_change, finger_dir, orientation_change, displacement, speed_str, flips_x, flips_y):
    #     dx, dy = displacement
    #     magnitude = np.linalg.norm(displacement)

    #     spatial_motion = "Stationary"
    #     articulation = "None"
    
        
    #     # -- 1. Start with checking for significant spatial movements --
    #     # check if possible several ==> append and have more than 1 in spatial motion
    #     if magnitude > 0.05: 
    #         # Calculate angle of displacement (-180 to 180 degrees)
    #         angle = np.degrees(np.arctan2(dy, dx))
            
    #         # Map angle to 8 points of the compass
    #         ## Directions are different due to webcam mirroring
    #         ## to adjust when using robot camera

    #         if -22.5 <= angle < 22.5: dir_str = "Right"
    #         elif 22.5 <= angle < 67.5: dir_str = "Down-Right"
    #         elif 67.5 <= angle < 112.5: dir_str = "Down"
    #         elif 112.5 <= angle < 157.5: dir_str = "Down-Left"
    #         elif angle >= 157.5 or angle < -157.5: dir_str = "Left"
    #         elif -157.5 <= angle < -112.5: dir_str = "Up-Left"
    #         elif -112.5 <= angle < -67.5: dir_str = "Up"
    #         elif -67.5 <= angle < -22.5: dir_str = "Up-Right"
    #         else: dir_str = "Unknown Direction"

    #         # Note: Because the camera is mirrored, depending on how your 
    #         # final UI works, you may need to flip "Left" and "Right" here.
            
    #         # if the string ends by Left
    #         if dir_str.endswith("Left"):
    #             dir_str = dir_str.replace("Left", "Right")
    #         elif dir_str.endswith("Right"):
    #             dir_str = dir_str.replace("Right", "Left")
            
    #         spatial_motion = f"{speed_str} Linear Translation towards {dir_str}"
        
    #     # -- 2. Check for wrist Rotations --
    #     elif orientation_change: 
    #         spatial_motion = f"{speed_str} Hand Rotation"
        
    #     # -- 3. Wave Detection (Oscillation) --
    #     # If the direction flips back and forth 2 or more times, it's a wave
    #     elif flips_x >= 2 or flips_y >= 2:
    #         spatial_motion = f"{speed_str} Oscillating Left & Right"

    #     # -- 4. Articulation --> only if the hand is relatively stable --
    #     if finger_change >= 4:
    #         articulation = f"{speed_str} All fingers bending towards the palm"
    #     if finger_change >= 1 and finger_change < 4:
    #         articulation = f"{speed_str} Bending Fingers"
        
            
    #     return spatial_motion, articulation