import numpy as np
from collections import deque

class TemporalGestureManager:
    def __init__(self, window_size=15):
        # Stores the distance between index tip and thumb tip over time
        self.history_dist = deque(maxlen=window_size)
        # Stores the detected static gestures to help smooth results
        self.gesture_history = deque(maxlen=window_size)
        self.window_size = window_size

    def update(self, hand_landmarks, static_gesture):
        """
        Processes new frame data and returns the temporal gesture.
        """
        # 1. Track Static Gesture for smoothing
        self.gesture_history.append(static_gesture)
        
        # 2. Track Distance for Pinching
        thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
        index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
        
        # Euclidean distance in normalized coordinates
        dist = np.linalg.norm(thumb_tip - index_tip)
        self.history_dist.append(dist)

        return self.analyze_motion()

    def analyze_motion(self):
        # Need enough frames to detect a trend
        if len(self.history_dist) < self.window_size:
            return None

        # --- PINCH DETECTION LOGIC ---
        # A pinch is characterized by the distance decreasing significantly 
        # then staying very small.
        start_dist = self.history_dist[0]
        end_dist = self.history_dist[-1]
        
        # Check if distance shrunk by at least 50% and ended very close
        if start_dist > 0.1 and end_dist < 0.03:
            return "Pinch Performed"

        # --- SMOOTHING LOGIC ---
        # Return the most frequent static gesture in the window (Mode)
        from collections import Counter
        most_common = Counter(self.gesture_history).most_common(1)[0][0]
        
        return most_common